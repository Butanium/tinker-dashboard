"""
Tinker Dashboard entry point.

Run with: uv run streamlit run app.py
"""

from src.dashboard import TinkerDashboard


def main():
    dashboard = TinkerDashboard()
    dashboard.display()


if __name__ == "__main__":
    main()
