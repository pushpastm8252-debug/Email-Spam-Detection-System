from pathlib import Path
import streamlit as st

def render_developers():
    """Render the two developer profiles for the Email Spam Detection app."""
    base_dir = Path(__file__).resolve().parent
    photo_dir = base_dir / "assets" / "developers"

    st.markdown("""
    <div style="
        background:linear-gradient(115deg,#0877F9,#123CBA 55%,#3216B9);
        border:1px solid #315BDE;border-radius:18px;
        padding:26px;margin-bottom:22px;">
        <h1 style="color:white;margin:0;">Meet Our Developers</h1>
        <p style="color:#E3EEFF;margin:8px 0 0;">
            The team behind the Email Spam Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)

    profiles = [
        {
            "name": "Pushpa Kumari",
            "title": "B.Tech Computer Science – AI & ML",
            "photo": photo_dir / "pushpa.jpg",
            "email": "pushpastm8252@gmail.com",
            "linkedin": "https://www.linkedin.com/in/pushpa-kumari-302184380",
            "github": "https://github.com/pushpastm8252-debug",
            "about": (
                "Computer Science student interested in Artificial Intelligence, "
                "Machine Learning and building practical software projects."
            ),
            "skills": ["Python", "Java", "C/C++", "AI/ML", "Streamlit", "SQL", "GitHub"],
            "contribution": "Machine learning, application development and project integration"
        },
        {
            "name": "Mohit Kumar",
            "title": "B.Tech – Artificial Intelligence & Machine Learning",
            "photo": photo_dir / "mohit.jpg",
            "email": "mk7469112@gmail.com",
            "linkedin": "https://www.linkedin.com/in/mohit-kumar-07to18",
            "github": "https://github.com/Mohitkumar-lan",
            "about": (
                "AI & ML student interested in Python development, embedded systems, "
                "IoT and practical technology projects."
            ),
            "skills": ["Python", "C", "C++", "SQL", "NumPy", "Pandas", "Arduino", "IoT"],
            "contribution": "Technical development and project collaboration"
        }
    ]

    col1, col2 = st.columns(2, gap="large")

    for col, person in zip([col1, col2], profiles):
        with col:
            st.markdown("""
            <div style="
                background:linear-gradient(145deg,#101F40,#0B1732);
                border:1px solid #304675;border-radius:18px;
                padding:20px;text-align:center;">
            </div>
            """, unsafe_allow_html=True)

            if person["photo"].exists():
                st.image(str(person["photo"]), width=210)
            else:
                st.warning(f"Profile photo not found: {person['photo']}")

            st.markdown(
                f"<h2 style='text-align:center;color:#F8FAFC'>{person['name']}</h2>"
                f"<p style='text-align:center;color:#9FC5FF'>{person['title']}</p>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div style='background:#101F40;border:1px solid #304675;"
                f"border-radius:14px;padding:16px;color:#E2E8F0;'>"
                f"<b>About</b><br>{person['about']}<br><br>"
                f"<b>Project Contribution</b><br>{person['contribution']}<br><br>"
                f"<b>Technical Skills</b><br>"
                + " ".join(
                    f"<span style='display:inline-block;background:#183B70;"
                    f"border:1px solid #3566A8;border-radius:18px;"
                    f"padding:5px 10px;margin:4px 3px;color:#EAF2FF;font-size:12px;'>"
                    f"{skill}</span>"
                    for skill in person["skills"]
                )
                + "</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:14px;">
                    <a href="mailto:{person['email']}"
                       style="background:#1455EF;color:white;padding:9px 13px;
                       border-radius:9px;text-decoration:none;font-weight:600;">
                       ✉ Email
                    </a>
                    <a href="{person['linkedin']}" target="_blank"
                       style="background:#0A66C2;color:white;padding:9px 13px;
                       border-radius:9px;text-decoration:none;font-weight:600;">
                       LinkedIn
                    </a>
                    <a href="{person['github']}" target="_blank"
                       style="background:#26364F;color:white;padding:9px 13px;
                       border-radius:9px;text-decoration:none;font-weight:600;">
                       GitHub
                    </a>
                </div>
                <p style="color:#AFC1E2;font-size:13px;margin-top:12px;">
                    {person['email']}
                </p>
                """,
                unsafe_allow_html=True
            )

    st.caption(
        "Instagram and portfolio links can be added when the developers provide them."
    )
