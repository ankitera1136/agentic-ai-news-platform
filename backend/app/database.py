from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.models import Base, Category, User
import os

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    import pathlib
    _here = pathlib.Path(__file__).resolve().parent
    for _parent in [_here, _here.parent, _here.parent.parent]:
        _env = _parent / ".env"
        if _env.exists():
            load_dotenv(str(_env), override=False)
            break
except ImportError:
    pass  # dotenv not installed — rely on externally set environment variables

# Database configuration
default_db_url = "sqlite:////tmp/tech_news.db" if (os.getenv("VERCEL") or os.getenv("NOW_REGION")) else "sqlite:///./tech_news.db"
DATABASE_URL = os.getenv("DATABASE_URL", default_db_url)

# Force psycopg2 dialect for PostgreSQL (natively supports sslmode=require)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL and "+pg8000" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
elif "+pg8000" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+pg8000", "+psycopg2", 1)

from sqlalchemy import event

connect_args = {"check_same_thread": False, "timeout": 30} if "sqlite" in DATABASE_URL else {}
pool_pre_ping = True if "sqlite" not in DATABASE_URL else False

engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    pool_pre_ping=pool_pre_ping
)

if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database, apply schema migrations, and create default categories"""
    Base.metadata.create_all(bind=engine)
    
    # Run automatic clustering migration (ensures new columns & tables exist on any host)
    try:
        try:
            from dedup.migrate import upgrade
        except ImportError:
            from ...dedup.migrate import upgrade
        upgrade()
    except Exception as e:
        print(f"[INIT DB WARNING] Story clustering migration note: {e}")

    # CareerRadar internal tables (article bodies, extraction ledger, signals)
    try:
        try:
            from career.migrate import upgrade as career_upgrade
        except ImportError:
            from ...career.migrate import upgrade as career_upgrade
        career_upgrade()
    except Exception as e:
        print(f"[INIT DB WARNING] CareerRadar migration note: {e}")

    # Student career profile table (CareerRadar step 2)
    try:
        try:
            from student_profile.migrate import upgrade as profile_upgrade
        except ImportError:
            from ...student_profile.migrate import upgrade as profile_upgrade
        profile_upgrade()
    except Exception as e:
        print(f"[INIT DB WARNING] Student profile migration note: {e}")

    # CareerRadar weekly briefs table (CareerRadar step 5)
    try:
        from backend.services.brief.migrate import upgrade as brief_upgrade
        brief_upgrade()
    except Exception as e:
        print(f"[INIT DB WARNING] Weekly brief migration note: {e}")

    # Auto-migration: Ensure columns exist
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE articles ADD COLUMN source VARCHAR DEFAULT 'TechNews'"))
            conn.commit()
    except Exception:
        pass
        
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE"))
            conn.commit()
    except Exception:
        pass
        
    try:
        from backend.app.models import User
        admin_email = os.getenv("ADMIN_EMAIL", "").lower().strip()
        if admin_email:
            db_admin = SessionLocal()
            admin_user = db_admin.query(User).filter(User.email == admin_email).first()
            if admin_user and not admin_user.is_admin:
                admin_user.is_admin = True
                db_admin.commit()
            db_admin.close()
    except Exception as e:
        print(f"Admin provisioning note: {e}")
    
    # Create or update default categories with rich 2026 taxonomy
    db = SessionLocal()
    try:
        default_categories = [
            {
                "name": "Artificial Intelligence",
                "description": "AI, machine learning, LLMs, and automation — research and applications",
                "keywords": ["AI", "artificial intelligence", "machine learning", "ML", "neural network",
                             "deep learning", "GPT", "LLM", "large language model", "automation", "robot",
                             "algorithm", "OpenAI", "ChatGPT", "Claude", "Anthropic", "Gemini", "DeepMind",
                             "Copilot", "reasoning", "model", "agent", "agentic", "Mistral", "Perplexity",
                             "DeepSeek", "Midjourney", "GenAI", "generative AI", "diffusion", "inference",
                             "prompt", "transformer", "synthetic", "fine-tuning", "RAG", "embedding",
                             "computer vision", "NLP", "natural language processing"]
            },
            {
                "name": "Indian Tech & Startups",
                "description": "Indian startup ecosystem, funding rounds, IT industry, and tech policy in India",
                "keywords": ["India", "Indian", "Bangalore", "Bengaluru", "Mumbai", "Delhi", "Hyderabad",
                             "Pune", "startup", "startups", "NASSCOM", "IT sector", "BPO", "ISRO",
                             "Digital India", "Make in India", "UPI", "Aadhaar", "fintech India",
                             "edtech India", "healthtech India", "Flipkart", "Zomato", "Swiggy", "Ola",
                             "Paytm", "Byju", "Meesho", "PhonePe", "Navi", "Razorpay", "CRED", "Zepto",
                             "unicorn India", "Series A India", "rupee", "SEBI", "RBI tech", "IIT", "IIM",
                             "Inc42", "YourStory", "MediaNama", "funding India", "valuation India"]
            },
            {
                "name": "Careers & Hiring in Tech",
                "description": "Job market, layoffs, hiring trends, internships, and salaries in tech",
                "keywords": ["hiring", "layoff", "layoffs", "job", "jobs", "career", "careers", "internship",
                             "internships", "salary", "salaries", "placement", "placements", "fresher",
                             "campus", "recruitment", "fired", "cut", "workforce", "employee", "employees",
                             "headcount", "job market", "tech jobs", "software engineer", "developer job",
                             "resume", "interview", "offer letter", "remote work", "hybrid", "work from home",
                             "H1B", "visa", "immigration tech", "retrenchment", "downsizing"]
            },
            {
                "name": "Open Source & Developer Tools",
                "description": "Open source projects, developer tools, frameworks, programming languages, and GitHub",
                "keywords": ["open source", "GitHub", "GitLab", "repository", "repo", "framework",
                             "library", "SDK", "API", "developer", "developers", "programming",
                             "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++",
                             "React", "Node.js", "FastAPI", "Django", "Docker", "Kubernetes",
                             "Linux", "terminal", "CLI", "DevOps", "CI/CD", "deployment",
                             "open-source", "contributing", "pull request", "package manager",
                             "npm", "pip", "release", "version", "Stack Overflow", "freeCodeCamp",
                             "tutorial", "documentation", "community", "contributor"]
            },
            {
                "name": "Cloud & Infrastructure",
                "description": "Cloud computing, serverless, databases, and backend infrastructure",
                "keywords": ["cloud", "AWS", "Azure", "Google Cloud", "GCP", "serverless", "Lambda",
                             "microservices", "database", "PostgreSQL", "MongoDB", "Redis", "Kafka",
                             "infrastructure", "DevOps", "platform", "SaaS", "PaaS", "IaaS",
                             "Cloudflare", "Vercel", "Netlify", "Kubernetes", "container", "Docker",
                             "Snowflake", "Databricks", "data engineering", "data pipeline",
                             "distributed systems", "scaling", "latency", "reliability", "uptime"]
            },
            {
                "name": "Cybersecurity",
                "description": "Security vulnerabilities, data breaches, privacy, and ethical hacking",
                "keywords": ["security", "breach", "hack", "hacker", "hackers", "cybersecurity",
                             "privacy", "data protection", "vulnerability", "vulnerabilities",
                             "malware", "ransomware", "encryption", "phishing", "exploit", "zero-day",
                             "leaked", "leak", "passwords", "CVE", "bug bounty", "penetration testing",
                             "ethical hacking", "VAPT", "infosec", "threat", "trojan", "surveillance",
                             "CERT", "patch", "authentication", "2FA", "MFA"]
            },
            {
                "name": "Startups & Funding",
                "description": "Global startup news, venture capital, funding rounds, and acquisitions",
                "keywords": ["startup", "funding", "venture capital", "VC", "investment", "seed",
                             "Series A", "Series B", "Series C", "IPO", "acquisition", "merger",
                             "valuation", "raise", "raised", "Y Combinator", "accelerator", "unicorn",
                             "angel", "capital", "founder", "founders", "pre-seed", "stealth",
                             "billion", "million", "deal", "exit", "investors", "backed"]
            },
            {
                "name": "Big Tech & Industry",
                "description": "News from major technology companies and IT industry trends",
                "keywords": ["Google", "Apple", "Microsoft", "Amazon", "Meta", "Facebook", "Tesla",
                             "Netflix", "Uber", "Twitter", "X", "Alphabet", "Qualcomm", "Intel",
                             "TSMC", "AMD", "YouTube", "AWS", "Azure", "Samsung", "Oracle", "IBM",
                             "ByteDance", "OpenAI", "Nvidia", "Infosys", "TCS", "Wipro", "HCL",
                             "Accenture", "Cognizant", "Tech Mahindra", "earnings", "quarterly"]
            },
            {
                "name": "Science & Research",
                "description": "Computer science research, scientific discoveries, space, and academia",
                "keywords": ["research", "paper", "study", "published", "arxiv", "conference", "NeurIPS",
                             "ICML", "CVPR", "ICLR", "IEEE", "ACM", "NASA", "ISRO", "space", "satellite",
                             "quantum", "physics", "biology", "climate", "science", "discovery",
                             "breakthrough", "experiment", "university", "academic", "PhD", "professor",
                             "benchmark", "dataset", "evaluation", "results"]
            },
            {
                "name": "Hardware & Chips",
                "description": "Semiconductors, CPUs, GPUs, chips, and computer hardware",
                "keywords": ["hardware", "chip", "chips", "semiconductor", "processor", "CPU", "GPU",
                             "ASML", "TSMC", "Intel", "AMD", "Qualcomm", "ARM", "RISC-V",
                             "RAM", "memory", "laptop", "PC", "server", "data center", "TPU",
                             "NPU", "silicon", "fabrication", "node", "nm", "wafer", "foundry",
                             "supply chain", "Nvidia GPU", "graphics card", "compute"]
            },
            {
                "name": "Product & Design",
                "description": "Product launches, UX/UI design, mobile apps, and consumer technology",
                "keywords": ["product", "launch", "released", "update", "feature", "app", "mobile",
                             "iOS", "Android", "iPhone", "design", "UX", "UI", "user experience",
                             "interface", "prototype", "beta", "v2", "redesign", "accessibility",
                             "dark mode", "performance", "review", "hands-on", "teardown",
                             "product manager", "roadmap", "feedback", "users", "subscription"]
            },
        ]
        
        for cat_data in default_categories:
            existing = db.query(Category).filter(Category.name == cat_data["name"]).first()
            if not existing:
                category = Category(**cat_data)
                db.add(category)
            else:
                existing.keywords = cat_data["keywords"]
                existing.description = cat_data["description"]
        
        db.commit()
        print("Updated category taxonomy with 2026 keywords")
    except Exception as e:
        print(f"Error initializing database categories: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()