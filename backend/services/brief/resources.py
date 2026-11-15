"""Static learning links for skills. Official documentation or tutorials only.

Links are looked up by canonical skill name (see backend/services/profile/skills.py). A skill
without an entry gets no link. The LLM never produces or sees URLs: code attaches these.
"""
from typing import Dict, List, Tuple

# canonical skill name -> ((title, url), ...) with 1 or 2 entries
LEARNING_RESOURCES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "Python": (("Python tutorial (python.org)", "https://docs.python.org/3/tutorial/"),),
    "JavaScript": (("JavaScript Guide (MDN)", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),),
    "TypeScript": (("TypeScript documentation", "https://www.typescriptlang.org/docs/"),),
    "React": (("Learn React (react.dev)", "https://react.dev/learn"),),
    "Node.js": (("Learn Node.js (nodejs.org)", "https://nodejs.org/en/learn"),),
    "Go": (
        ("A Tour of Go", "https://go.dev/tour/"),
        ("Go documentation", "https://go.dev/doc/"),
    ),
    "Rust": (("The Rust Programming Language (book)", "https://doc.rust-lang.org/book/"),),
    "Java": (("Learn Java (dev.java)", "https://dev.java/learn/"),),
    "SQL": (("The SQL Language (PostgreSQL docs)", "https://www.postgresql.org/docs/current/tutorial-sql.html"),),
    "PostgreSQL": (("PostgreSQL tutorial", "https://www.postgresql.org/docs/current/tutorial.html"),),
    "MongoDB": (("MongoDB getting started", "https://www.mongodb.com/docs/manual/tutorial/getting-started/"),),
    "Docker": (("Docker get started", "https://docs.docker.com/get-started/"),),
    "Kubernetes": (("Learn Kubernetes basics", "https://kubernetes.io/docs/tutorials/kubernetes-basics/"),),
    "AWS": (("AWS getting started", "https://aws.amazon.com/getting-started/"),),
    "Azure": (("Azure documentation (Microsoft Learn)", "https://learn.microsoft.com/en-us/azure/"),),
    "GCP": (("Google Cloud documentation", "https://cloud.google.com/docs"),),
    "Machine Learning": (
        ("Machine Learning Crash Course (Google)", "https://developers.google.com/machine-learning/crash-course"),
        ("scikit-learn user guide", "https://scikit-learn.org/stable/user_guide.html"),
    ),
    "Deep Learning": (("PyTorch tutorials", "https://pytorch.org/tutorials/"),),
    "PyTorch": (("PyTorch tutorials", "https://pytorch.org/tutorials/"),),
    "TensorFlow": (("TensorFlow tutorials", "https://www.tensorflow.org/tutorials"),),
    "scikit-learn": (("scikit-learn user guide", "https://scikit-learn.org/stable/user_guide.html"),),
    "Pandas": (("pandas getting started", "https://pandas.pydata.org/docs/getting_started/index.html"),),
    "NumPy": (("NumPy absolute beginners guide", "https://numpy.org/doc/stable/user/absolute_beginners.html"),),
    "HTML": (("Learn HTML (MDN)", "https://developer.mozilla.org/en-US/docs/Learn/HTML"),),
    "CSS": (("Learn CSS (MDN)", "https://developer.mozilla.org/en-US/docs/Learn/CSS"),),
    "Tailwind CSS": (("Tailwind CSS documentation", "https://tailwindcss.com/docs"),),
    "Git": (("Git documentation", "https://git-scm.com/doc"),),
    "GitHub": (("GitHub get started", "https://docs.github.com/en/get-started"),),
    "Django": (("Django tutorial", "https://docs.djangoproject.com/en/stable/intro/tutorial01/"),),
    "Flask": (("Flask tutorial", "https://flask.palletsprojects.com/en/stable/tutorial/"),),
    "FastAPI": (("FastAPI tutorial", "https://fastapi.tiangolo.com/tutorial/"),),
    "Angular": (("Angular tutorials", "https://angular.dev/tutorials"),),
    "Vue.js": (("Vue.js tutorial", "https://vuejs.org/tutorial/"),),
    "Next.js": (("Learn Next.js", "https://nextjs.org/learn"),),
    "Express.js": (("Express getting started", "https://expressjs.com/en/starter/installing.html"),),
    "Spring Boot": (("Spring guides", "https://spring.io/guides"),),
    "GraphQL": (("Learn GraphQL", "https://graphql.org/learn/"),),
    "Redis": (("Redis documentation", "https://redis.io/docs/latest/"),),
    "Kotlin": (("Kotlin getting started", "https://kotlinlang.org/docs/getting-started.html"),),
    "Swift": (("Swift documentation", "https://www.swift.org/documentation/"),),
    "C#": (("A tour of C# (Microsoft Learn)", "https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/"),),
}


def links_for(skill: str) -> List[Dict[str, str]]:
    """Learning links for a canonical skill name; an empty list when there is no entry."""
    return [{"title": title, "url": url} for title, url in LEARNING_RESOURCES.get(skill, ())]
