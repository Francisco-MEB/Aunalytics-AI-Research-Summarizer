# RAPTOR QA System - User Authentication

##  Multi-User Setup

Each user has their own isolated documents and cannot see other users' data. This is enforced by Row-Level Security (RLS) in the database.

## Quick Start

### 1. Register a New Account

```powershell
python auth.py register <username> <password> [email]
```

Example:
```powershell
python auth.py register alice mypassword123 alice@example.com
```

This creates a unique User ID for you and logs you in automatically.

### 2. Login (Subsequent Sessions)

```powershell
python auth.py login <username> <password>
```

Example:
```powershell
python auth.py login alice mypassword123
```

Your session is saved locally in `.session` file.

### 3. Use the QA System

Once logged in, you can use all commands without passing user_id:

```powershell
# Ask questions
python raptor_qa_incremental.py

# Add documents
python add_document.py data/sample.txt

# Delete documents
python delete_document.py <doc_id>
```

## Commands Reference

### Authentication Commands

```powershell
# Register new user
python auth.py register <username> <password> [email]

# Login
python auth.py login <username> <password>

# List all users (admin)
python auth.py list
```

### QA System Commands

```powershell
# Start QA session (requires login)
python raptor_qa_incremental.py

# Then type:
#   list              - See your documents
#   diagnose          - DB stats
#   summarize_all     - Overview of all docs
#   <your question>   - Ask anything
#   quit              - Exit
```

### Document Management

```powershell
# Add a document (requires login)
python add_document.py path/to/file.pdf

# Delete a document (requires login)
# First get doc_id from 'list' command in QA system
python delete_document.py <doc_id>
```

## Security Features

 **Password Hashing**: Passwords stored as SHA-256 hashes
 **Row-Level Security**: Users can only access their own documents
 **Session Management**: Login persists across commands
 **User Isolation**: Each user_id filters all database queries

## Files Created

- `.auth_users.json` - User database (username, hashed password, user_id)
- `.session` - Current login session (username, user_id)

**️ Add these to `.gitignore`:**
```
.auth_users.json
.session
```

## Multi-User Example

```powershell
# Alice registers and adds documents
python auth.py register alice pass123
python add_document.py research_paper.pdf
python raptor_qa_incremental.py
# Alice types: list
# Shows only Alice's documents

# Bob registers and adds documents
python auth.py register bob pass456
python add_document.py different_doc.pdf
python raptor_qa_incremental.py
# Bob types: list
# Shows only Bob's documents (not Alice's!)
```

## Migration from Old System

If you were using the old system with hardcoded user_id:

```powershell
# 1. Register with a username
python auth.py register myusername mypassword

# 2. Note your new user_id (shown after registration)

# 3. Update your existing data in Supabase:
UPDATE chunks SET user_id = '<new-user-id>' WHERE user_id = '<old-user-id>';
UPDATE tree_nodes SET user_id = '<new-user-id>' WHERE user_id = '<old-user-id>';
UPDATE chunk_to_leaf SET user_id = '<new-user-id>' WHERE user_id = '<old-user-id>';
```

## FAQ

**Q: Where is my user_id stored?**  
A: In `.session` file after login, and in `.auth_users.json` database.

**Q: Can I use the old system with hardcoded user_id?**  
A: No, the CLI tools now require authentication. This ensures proper multi-user support.

**Q: What if I forget my password?**  
A: Currently no password reset. You can manually edit `.auth_users.json` to set a new hash, or create a new account.

**Q: Is this production-ready authentication?**  
A: This is a simple file-based auth for local/small deployments. For production, use proper authentication (OAuth, JWT, etc.) with a real user database.
