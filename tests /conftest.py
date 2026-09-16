"""Shared pytest fixtures for SmartHire tests."""
from __future__ import annotations

import pandas as pd
import pytest


_CATEGORIES = [
    "Data Science", "Java Developer", "Testing", "Python Developer",
    "DevOps Engineer", "Database", "HR", "Business Analyst",
]


@pytest.fixture()
def tiny_resumes() -> pd.DataFrame:
    """Labelled resume dataset — 2 samples per category (8 categories × 2 = 16 rows)."""
    rows = []
    templates = {
        "Data Science": [
            "python machine learning scikit-learn pandas numpy statistics data analysis",
            "data scientist sklearn regression classification feature engineering sql",
        ],
        "Java Developer": [
            "java spring boot hibernate rest api postgresql docker microservices",
            "java developer junit maven spring mvc mysql backend enterprise application",
        ],
        "Testing": [
            "selenium java testng page object model automation testing ci cd jenkins",
            "qa engineer manual testing jira bug tracking regression api testing postman",
        ],
        "Python Developer": [
            "python django fastapi rest api postgresql redis celery docker backend",
            "python flask sqlalchemy postgresql unit testing pytest clean code",
        ],
        "DevOps Engineer": [
            "aws kubernetes docker terraform ci cd linux jenkins github actions devops",
            "devops engineer helm argo cd prometheus grafana sre kubernetes aws",
        ],
        "Database": [
            "sql oracle postgresql dba performance tuning backup recovery stored procedures",
            "database administrator mysql replication indexing query optimisation",
        ],
        "HR": [
            "recruitment onboarding hrms payroll employee relations talent acquisition",
            "hr generalist performance management training development workforce planning",
        ],
        "Business Analyst": [
            "requirements gathering stakeholder communication sql excel tableau use cases",
            "business analyst agile user stories process mapping bpmn jira product owner",
        ],
    }
    for category in _CATEGORIES:
        for text in templates[category]:
            rows.append({"category": category, "text": text})
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_jobs() -> pd.DataFrame:
    """Minimal job corpus with 6 listings covering different domains."""
    rows = [
        {
            "title": "data scientist",
            "company": "techco",
            "location": "remote",
            "skills": "python machine learning scikit-learn statistics sql",
            "description": "build ml models and deploy data products for business decisions",
            "experience": "1-3 years",
            "document": "data scientist python machine learning scikit-learn statistics sql build ml models",
        },
        {
            "title": "java developer",
            "company": "softcorp",
            "location": "bangalore",
            "skills": "java spring boot hibernate postgresql docker",
            "description": "design and build enterprise web applications using java spring",
            "experience": "2-4 years",
            "document": "java developer spring boot hibernate postgresql docker enterprise web applications",
        },
        {
            "title": "devops engineer",
            "company": "cloudco",
            "location": "remote",
            "skills": "aws kubernetes docker terraform ci cd linux",
            "description": "maintain cloud infrastructure and ci cd pipelines for engineering teams",
            "experience": "2-5 years",
            "document": "devops engineer aws kubernetes docker terraform ci cd linux cloud infrastructure pipelines",
        },
        {
            "title": "python developer",
            "company": "apico",
            "location": "pune",
            "skills": "python django fastapi postgresql redis docker",
            "description": "design rest apis and reliable backend services using python django",
            "experience": "1-3 years",
            "document": "python developer django fastapi postgresql redis docker rest api backend services",
        },
        {
            "title": "hr specialist",
            "company": "peopleco",
            "location": "delhi",
            "skills": "recruitment onboarding hrms payroll communication",
            "description": "manage recruitment employee onboarding hrms and stakeholder communication",
            "experience": "1-3 years",
            "document": "hr specialist recruitment onboarding hrms payroll communication employee relations",
        },
        {
            "title": "business analyst",
            "company": "strategyco",
            "location": "chennai",
            "skills": "sql excel tableau requirements stakeholder communication",
            "description": "gather requirements produce business analysis and reporting dashboards",
            "experience": "1-3 years",
            "document": "business analyst sql excel tableau requirements stakeholder analysis reporting",
        },
    ]
    return pd.DataFrame(rows)
