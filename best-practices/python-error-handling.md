# Python Error Handling Best Practices

## Principle: Be Specific with Exception Handling

### Bad Practice: Catching All Exceptions

```python
def read_config(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except:  # Too broad - catches everything including KeyboardInterrupt
        return {}
```

**Why this is bad:**
- Catches system exceptions like `KeyboardInterrupt` and `SystemExit`
- Hides bugs and makes debugging difficult
- Doesn't distinguish between different failure modes
- Silent failures can lead to incorrect program behavior

### Good Practice: Catch Specific Exceptions

```python
def read_config(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {filename} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        raise
    except PermissionError:
        logger.error(f"Permission denied reading {filename}")
        raise
```

**Why this is good:**
- Catches only expected exceptions
- Different error types are handled appropriately
- Provides clear logging for debugging
- Allows unexpected errors to propagate
- System exceptions are not caught

## Principle: Don't Silence Errors Without Good Reason

### Bad Practice: Empty Exception Handler

```python
def save_user_preferences(user_id, prefs):
    try:
        db.update(user_id, prefs)
    except:
        pass  # Silent failure - user thinks preferences were saved
```

**Why this is bad:**
- User receives no feedback about the failure
- No logging for debugging
- Data loss without notification
- Impossible to diagnose issues in production

### Good Practice: Handle or Propagate with Context

```python
def save_user_preferences(user_id, prefs):
    try:
        db.update(user_id, prefs)
    except DatabaseConnectionError as e:
        logger.error(f"Failed to save preferences for user {user_id}: {e}")
        raise UserPreferencesError("Unable to save preferences. Please try again.") from e
```

**Why this is good:**
- Logs the actual error for debugging
- Provides user-friendly error message
- Preserves exception chain with `from e`
- Allows calling code to handle the error appropriately
