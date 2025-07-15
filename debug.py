from flask import Flask, session, jsonify

def add_debug_routes(app):
    @app.route('/debug/session')
    def debug_session():
        """
        A debugging route to view the contents of the session.
        IMPORTANT: DISABLE THIS IN PRODUCTION!
        """
        # Session data (safely converted to dict for JSON response)
        session_data = dict(session)
        
        # Check for specific keys
        has_username = 'username' in session
        has_last_prediction = 'last_prediction' in session
        
        # Prepare a user-friendly response
        response = {
            'has_username': has_username,
            'has_last_prediction': has_last_prediction,
            'session_keys': list(session.keys()),
            'session_data': session_data
        }
        
        return jsonify(response)

# To use this, add the following to your app.py:
# from debug import add_debug_routes
# add_debug_routes(app)
