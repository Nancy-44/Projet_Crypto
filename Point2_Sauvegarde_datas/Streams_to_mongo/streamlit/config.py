"""

Config file for Streamlit App

"""

from .member import Member


TITLE = "CryptoBot"

TEAM_MEMBERS = [
    Member(
        name="Nancy Frémont",
        linkedin_url="https://www.linkedin.com/in/",
        github_url="https://github.com/Nancy-44",
    ),
    Member(
        name="Phillipe Kirstetter-Fender",
        linkedin_url="https://www.linkedin.com/in/",
        github_url="https://github.com/PhilippeK-F",
    ),
    Member(
        name="Florent Rigal",
        linkedin_url="https://www.linkedin.com/in/",
        github_url="https://github.com/",
    ),
    Member(
        name="Thomas Saliou",
        linkedin_url="https://www.linkedin.com/in/thomas-saliou-6371558b/",
        github_url="https://github.com/7omate",
    )
]

PROMOTION = "Promotion Bootcamp Data Engineer - Sep 2025"
