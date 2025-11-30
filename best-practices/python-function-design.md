# Python Function Design Best Practices

## Principle: Functions Should Do One Thing Well

### Bad Practice: Function Does Too Much

```python
def process_user_registration(email, password, name):
    # Validate email
    if '@' not in email:
        return None, "Invalid email"

    # Hash password
    hashed = hashlib.sha256(password.encode()).hexdigest()

    # Create user in database
    user = db.create_user(email, hashed, name)

    # Send welcome email
    smtp.send(email, "Welcome!", "Thanks for joining!")

    # Log analytics
    analytics.track('user_registered', {'email': email})

    # Update cache
    cache.set(f'user:{user.id}', user)

    return user, None
```

**Why this is bad:**
- Violates Single Responsibility Principle
- Hard to test each piece independently
- Difficult to reuse individual operations
- Changes to one aspect require modifying this function
- Hard to understand at a glance

### Good Practice: Single Responsibility Functions

```python
def validate_email(email):
    if '@' not in email or '.' not in email.split('@')[1]:
        raise ValueError("Invalid email format")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password_hash, name):
    return db.create_user(email, password_hash, name)

def send_welcome_email(user):
    smtp.send(user.email, "Welcome!", "Thanks for joining!")

def track_registration(user):
    analytics.track('user_registered', {'email': user.email})

def process_user_registration(email, password, name):
    validate_email(email)
    password_hash = hash_password(password)
    user = create_user(email, password_hash, name)
    send_welcome_email(user)
    track_registration(user)
    return user
```

**Why this is good:**
- Each function has a single, clear purpose
- Easy to test each function independently
- Functions can be reused in other contexts
- Changes are isolated to specific functions
- Reads like a high-level description of the process

## Principle: Avoid Flag Arguments

### Bad Practice: Boolean Flag Changes Behavior

```python
def calculate_price(item, include_tax):
    base_price = item.price
    if include_tax:
        return base_price * 1.08
    else:
        return base_price
```

**Why this is bad:**
- Function does two different things based on a flag
- Calling code is unclear: `calculate_price(item, True)` - what does True mean?
- Often a sign that you need two functions
- Harder to extend (what if you need more tax options?)

### Good Practice: Separate Functions or Explicit Parameters

```python
def calculate_base_price(item):
    return item.price

def calculate_price_with_tax(item, tax_rate=0.08):
    base_price = calculate_base_price(item)
    return base_price * (1 + tax_rate)
```

**Why this is good:**
- Function names clearly indicate what they do
- No ambiguity when reading the code
- Easy to add more tax calculation options
- Each function has a clear contract

## Principle: Return Early to Reduce Nesting

### Bad Practice: Deep Nesting

```python
def process_payment(user, amount):
    if user is not None:
        if user.is_active:
            if amount > 0:
                if user.balance >= amount:
                    user.balance -= amount
                    return {"success": True, "new_balance": user.balance}
                else:
                    return {"success": False, "error": "Insufficient funds"}
            else:
                return {"success": False, "error": "Invalid amount"}
        else:
            return {"success": False, "error": "User not active"}
    else:
        return {"success": False, "error": "User not found"}
```

**Why this is bad:**
- Deep nesting is hard to read
- Happy path is buried at the deepest level
- Hard to see all the error conditions
- Difficult to maintain

### Good Practice: Guard Clauses

```python
def process_payment(user, amount):
    if user is None:
        return {"success": False, "error": "User not found"}

    if not user.is_active:
        return {"success": False, "error": "User not active"}

    if amount <= 0:
        return {"success": False, "error": "Invalid amount"}

    if user.balance < amount:
        return {"success": False, "error": "Insufficient funds"}

    user.balance -= amount
    return {"success": True, "new_balance": user.balance}
```

**Why this is good:**
- Flat structure is easier to read
- Error conditions are clear and visible
- Happy path is at the bottom, not nested
- Easy to add or modify validation rules
- Each guard clause is independent
