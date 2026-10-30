"""
Vercel Python entry point for the Rudo chatbot Flask application.
This wrapper allows Vercel to properly locate and initialize the Flask app.
"""
from main import app

__all__ = ['app']
