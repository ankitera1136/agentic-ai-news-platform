"""Deterministic skill-name canonicalization.

A small curated map folds common aliases into one display name so that a student's
"js", "JavaScript" and "javascript" all become "JavaScript". Skills not in the map keep
their cleaned original spelling.
"""
import re
from typing import Dict

# canonical display name -> aliases (matched case-insensitively, whitespace-collapsed)
_CANONICAL_SKILLS: Dict[str, tuple] = {
    "JavaScript": ("js", "javascript", "java script", "ecmascript", "es6"),
    "TypeScript": ("ts", "typescript"),
    "Python": ("py", "python", "python3", "python 3", "python2"),
    "Java": ("java", "core java"),
    "C": ("c",),
    "C++": ("c++", "cpp", "cplusplus"),
    "C#": ("c#", "csharp", "c sharp"),
    "Go": ("go", "golang"),
    "Rust": ("rust", "rustlang"),
    "Kotlin": ("kotlin",),
    "Swift": ("swift",),
    "PHP": ("php",),
    "Ruby": ("ruby",),
    "SQL": ("sql",),
    "HTML": ("html", "html5"),
    "CSS": ("css", "css3"),
    "Tailwind CSS": ("tailwind", "tailwindcss", "tailwind css"),
    "React": ("react", "reactjs", "react.js", "react js"),
    "Next.js": ("nextjs", "next.js", "next js"),
    "Angular": ("angular", "angularjs", "angular.js"),
    "Vue.js": ("vue", "vuejs", "vue.js", "vue js"),
    "Node.js": ("node", "nodejs", "node.js", "node js"),
    "Express.js": ("expressjs", "express.js"),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi", "fast api"),
    "Spring Boot": ("spring boot", "springboot"),
    "GraphQL": ("graphql",),
    "REST APIs": ("rest", "rest api", "rest apis", "restful api", "restful apis"),
    "Machine Learning": ("ml", "machine learning"),
    "Deep Learning": ("deep learning",),
    "Natural Language Processing": ("nlp", "natural language processing"),
    "Computer Vision": ("computer vision",),
    "Data Science": ("data science",),
    "Data Structures and Algorithms": ("dsa", "data structures and algorithms", "data structures & algorithms"),
    "TensorFlow": ("tensorflow", "tensor flow"),
    "PyTorch": ("pytorch", "py torch"),
    "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Docker": ("docker",),
    "AWS": ("aws", "amazon web services"),
    "GCP": ("gcp", "google cloud", "google cloud platform"),
    "Azure": ("azure", "microsoft azure"),
    "PostgreSQL": ("postgres", "postgresql", "postgre sql", "psql"),
    "MySQL": ("mysql", "my sql"),
    "MongoDB": ("mongo", "mongodb", "mongo db"),
    "Redis": ("redis",),
    "Git": ("git",),
    "GitHub": ("github", "git hub"),
    "Linux": ("linux",),
    "CI/CD": ("ci/cd", "cicd", "ci cd"),
}


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in _CANONICAL_SKILLS.items():
    _ALIAS_TO_CANONICAL[_key(_canonical)] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_key(_alias)] = _canonical


def canonicalize_skill(raw: str) -> str:
    """Return the canonical name for a known skill alias, else the input unchanged.

    The caller is expected to pass already-cleaned text (see text.clean_text).
    """
    return _ALIAS_TO_CANONICAL.get(_key(raw), raw)
