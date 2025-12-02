from flask import Flask, jsonify
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from routes.health import health_bp
from routes.diagnose import diagnose_bp
from utils.error_handler import setup_error_handlers

def create_app():
    """Flask application factory."""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(diagnose_bp)
    
    # Error handlers
    setup_error_handlers(app)
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )