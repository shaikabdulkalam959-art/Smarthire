from __future__ import annotations

import re
from sklearn.metrics.pairwise import cosine_similarity

# Expanded skill vocabulary covering all 25 resume categories
SKILL_VOCABULARY = {
    # Data Science / ML
    "python", "r", "sql", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "statistics", "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "xgboost", "lightgbm", "random forest", "logistic regression", "linear regression",
    "clustering", "classification", "regression", "nlp", "computer vision",
    "jupyter", "scipy", "statsmodels", "feature engineering", "data analysis",
    "tableau", "power bi", "excel", "looker", "data visualization",

    # DevOps / Cloud / Infrastructure
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "ci cd", "github actions", "linux", "bash", "helm", "prometheus", "grafana",
    "nginx", "argocd", "datadog", "cloudformation", "sre",

    # Backend / Web Dev
    "django", "fastapi", "flask", "spring boot", "spring", "nodejs", "express",
    "rest api", "graphql", "grpc", "postgresql", "mysql", "mongodb", "redis",
    "kafka", "rabbitmq", "celery", "microservices", "docker compose",

    # Frontend / Web Design
    "javascript", "typescript", "react", "angular", "vue", "html", "css",
    "sass", "tailwind", "figma", "adobe xd", "bootstrap", "webpack", "nextjs",

    # Java
    "java", "spring mvc", "hibernate", "maven", "gradle", "junit", "testng", "jpa",
    "jvm", "jsp", "servlet", "struts", "jee", "ejb",

    # .NET
    "c#", ".net", "asp.net", "blazor", "entity framework", "wpf", "xamarin", "maui",
    "azure devops", "signalr",

    # Testing / QA
    "selenium", "playwright", "cypress", "appium", "postman", "jmeter", "gatling",
    "rest assured", "cucumber", "robot framework", "katalon",
    "manual testing", "automation testing", "performance testing", "api testing",

    # Database / ETL
    "oracle", "sql server", "db2", "cassandra", "elasticsearch", "neo4j", "snowflake",
    "redshift", "bigquery", "hive", "hbase", "hdfs", "spark", "airflow",
    "ssis", "informatica", "talend", "etl", "data warehouse", "dbt",

    # Hadoop / Big Data
    "hadoop", "mapreduce", "pig", "oozie", "sqoop", "flume", "yarn",
    "cloudera", "hortonworks", "pyspark",

    # Blockchain
    "solidity", "ethereum", "web3", "smart contracts", "hyperledger", "nft",
    "defi", "blockchain", "cryptography",

    # SAP
    "sap", "abap", "sap hana", "sap fiori", "sap s4hana", "sap bw", "sap mm",
    "sap sd", "sap fico", "sap basis",

    # Network Security
    "firewall", "vpn", "ids", "ips", "siem", "penetration testing", "ethical hacking",
    "nessus", "burp suite", "wireshark", "cisco", "palo alto", "splunk",
    "vulnerability assessment", "soc", "incident response",

    # HR
    "recruitment", "onboarding", "hrms", "payroll", "performance management",
    "talent acquisition", "employee relations", "workforce planning", "training",

    # Business Analysis / PMO
    "requirements gathering", "stakeholder management", "jira", "confluence",
    "agile", "scrum", "kanban", "bpmn", "use cases", "ms project", "pmp",
    "prince2", "risk management", "business analysis",

    # Engineering domains
    "autocad", "solidworks", "catia", "ansys", "matlab", "simulink", "etap",
    "staad pro", "plc", "scada", "lean", "six sigma",
    "gd&t", "fea", "hvac", "mechanical design",

    # Soft / Certification
    "git", "communication", "leadership", "project management",
}

# Single compiled regex: match longest terms first with word boundary lookahead/behind
_SKILL_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9#+])(?:"
    + "|".join(re.escape(s) for s in sorted(SKILL_VOCABULARY, key=len, reverse=True))
    + r")(?![a-zA-Z0-9#+])",
    re.IGNORECASE,
)


def extract_skills(text: str) -> set[str]:
    """Extract known technical and domain skills from text in a single fast regex pass."""
    if not text:
        return set()
    return {m.group(0).lower() for m in _SKILL_PATTERN.finditer(str(text))}


def skill_overlap(resume_text: str, job_text: str) -> float:
    """Fraction of job skills that appear in the resume (Precision metric)."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    return len(resume_skills & job_skills) / max(len(job_skills), 1)


def pair_features(resume_text: str, job_text: str, vectorizer) -> list[float]:
    """Return [cosine_similarity, skill_overlap] feature vector for a (resume, job) pair."""
    vectors = vectorizer.transform([resume_text, job_text])
    cos = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
    overlap = skill_overlap(resume_text, job_text)
    return [cos, overlap]
