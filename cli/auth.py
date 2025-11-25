"""
Simple authentication system for RAPTOR QA
Stores user credentials and manages sessions
"""
import os
import json
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Dict

class AuthManager:
    def __init__(self, auth_file: str = ".auth_users.json"):
        """Initialize auth manager with local user database"""
        self.auth_file = Path(auth_file)
        self.users = self._load_users()
        self.current_user = None
    
    def _load_users(self) -> Dict:
        """Load users from file"""
        if self.auth_file.exists():
            with open(self.auth_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_users(self):
        """Save users to file"""
        with open(self.auth_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password with SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username: str, password: str, email: str = None) -> str:
        """
        Register a new user
        
        Returns:
            user_id (UUID string)
        """
        if username in self.users:
            raise ValueError(f"Username '{username}' already exists")
        
        user_id = str(uuid.uuid4())
        self.users[username] = {
            'user_id': user_id,
            'password_hash': self._hash_password(password),
            'email': email,
            'created_at': str(uuid.uuid1().time)
        }
        self._save_users()
        
        print(f" User '{username}' registered successfully!")
        print(f"   User ID: {user_id}")
        return user_id
    
    def login(self, username: str, password: str) -> Optional[str]:
        """
        Login user
        
        Returns:
            user_id if successful, None otherwise
        """
        if username not in self.users:
            print(f"User '{username}' not found")
            return None
        
        user = self.users[username]
        if user['password_hash'] != self._hash_password(password):
            print("Invalid password")
            return None
        
        self.current_user = {
            'username': username,
            'user_id': user['user_id']
        }
        
        print(f"Logged in as {username}")
        return user['user_id']
    
    def get_current_user_id(self) -> Optional[str]:
        """Get current logged-in user's ID"""
        if self.current_user:
            return self.current_user['user_id']
        return None
    
    def list_users(self):
        """List all registered users (admin function)"""
        if not self.users:
            print("No users registered")
            return
        
        print("\n Registered Users:")
        for username, data in self.users.items():
            print(f"  - {username}")
            print(f"    User ID: {data['user_id']}")
            if data.get('email'):
                print(f"    Email: {data['email']}")
    
    def save_session(self, session_file: str = ".session"):
        """Save current session to file"""
        if not self.current_user:
            print("No user logged in")
            return
        
        with open(session_file, 'w') as f:
            json.dump(self.current_user, f)
        print(f"Session saved")
    
    def load_session(self, session_file: str = ".session") -> Optional[str]:
        """Load session from file"""
        session_path = Path(session_file)
        if not session_path.exists():
            return None
        
        with open(session_file, 'r') as f:
            self.current_user = json.load(f)
        
        print(f"Restored session for {self.current_user['username']}")
        return self.current_user['user_id']
    
    def logout(self, session_file: str = ".session"):
        """Logout current user"""
        session_path = Path(session_file)
        if session_path.exists():
            session_path.unlink()
        
        if self.current_user:
            username = self.current_user['username']
            self.current_user = None
            print(f"Logged out {username}")
        else:
            print("No active session")


def main():
    """CLI for user management"""
    import sys
    
    auth = AuthManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python auth.py register <username> <password> [email]")
        print("  python auth.py login <username> <password>")
        print("  python auth.py logout")
        print("  python auth.py list")
        print("\nExamples:")
        print("  python auth.py register john mypassword john@example.com")
        print("  python auth.py login john mypassword")
        print("  python auth.py logout")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "register":
        if len(sys.argv) < 4:
            print("Usage: python auth.py register <username> <password> [email]")
            sys.exit(1)
        
        username = sys.argv[2]
        password = sys.argv[3]
        email = sys.argv[4] if len(sys.argv) > 4 else None
        
        try:
            user_id = auth.register(username, password, email)
            auth.login(username, password)
            auth.save_session()
            print(f"\nYou can now use the QA system!")
            print(f"   Run: python raptor_qa_incremental.py")
        except ValueError as e:
            print(f"{e}")
    
    elif command == "login":
        if len(sys.argv) < 4:
            print("Usage: python auth.py login <username> <password>")
            sys.exit(1)
        
        username = sys.argv[2]
        password = sys.argv[3]
        
        user_id = auth.login(username, password)
        if user_id:
            auth.save_session()
            print(f"\nYou can now use the QA system!")
            print(f"   Run: python raptor_qa_incremental.py")
    
    elif command == "list":
        auth.list_users()
    
    elif command == "logout":
        auth.load_session()
        auth.logout()
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: register, login, logout, list")


if __name__ == "__main__":
    main()
