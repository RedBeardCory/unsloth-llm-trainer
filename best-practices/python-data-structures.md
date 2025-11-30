# Python Data Structures Best Practices

## Principle: Use the Right Data Structure for the Job

### Bad Practice: Using Lists for Membership Testing

```python
def find_duplicate_users(user_ids):
    seen = []  # List has O(n) lookup time
    duplicates = []

    for user_id in user_ids:
        if user_id in seen:  # Slow for large lists
            duplicates.append(user_id)
        else:
            seen.append(user_id)

    return duplicates
```

**Why this is bad:**
- `in` operator on lists is O(n)
- For 10,000 items, this becomes very slow
- Poor performance at scale
- Wrong data structure for the operation

### Good Practice: Use Sets for Membership Testing

```python
def find_duplicate_users(user_ids):
    seen = set()  # Set has O(1) lookup time
    duplicates = []

    for user_id in user_ids:
        if user_id in seen:  # Fast constant-time lookup
            duplicates.append(user_id)
        else:
            seen.add(user_id)

    return duplicates
```

**Why this is good:**
- Sets provide O(1) membership testing
- Scales well to large datasets
- Correct data structure for the operation
- More efficient memory usage for lookups

## Principle: Use Dictionary Lookup Instead of if/elif Chains

### Bad Practice: Long if/elif Chain

```python
def get_discount_rate(customer_type):
    if customer_type == 'bronze':
        return 0.05
    elif customer_type == 'silver':
        return 0.10
    elif customer_type == 'gold':
        return 0.15
    elif customer_type == 'platinum':
        return 0.20
    elif customer_type == 'diamond':
        return 0.25
    else:
        return 0.0
```

**Why this is bad:**
- Verbose and repetitive
- Hard to maintain as list grows
- Not data-driven
- Difficult to modify discount rates
- Slow for many conditions

### Good Practice: Dictionary Lookup

```python
DISCOUNT_RATES = {
    'bronze': 0.05,
    'silver': 0.10,
    'gold': 0.15,
    'platinum': 0.20,
    'diamond': 0.25,
}

def get_discount_rate(customer_type):
    return DISCOUNT_RATES.get(customer_type, 0.0)
```

**Why this is good:**
- Data is separated from logic
- Easy to modify rates
- O(1) lookup time
- More concise and readable
- Easy to load from configuration/database

## Principle: Use List Comprehensions for Transformations

### Bad Practice: Manual List Building

```python
def get_active_user_emails(users):
    emails = []
    for user in users:
        if user.is_active:
            emails.append(user.email.lower())
    return emails
```

**Why this is bad:**
- More verbose than necessary
- Multiple lines for a simple transformation
- Not idiomatic Python
- Harder to see the intent at a glance

### Good Practice: List Comprehension

```python
def get_active_user_emails(users):
    return [user.email.lower() for user in users if user.is_active]
```

**Why this is good:**
- Concise and readable
- Idiomatic Python
- Intent is immediately clear
- Often faster than manual loops
- Single expression

**Note:** For complex transformations, explicit loops are clearer:

```python
# When logic is complex, explicit loop is better
def process_users(users):
    results = []
    for user in users:
        # Complex multi-step processing
        normalized_name = user.name.strip().title()
        if validate_email(user.email):
            age = calculate_age(user.birth_date)
            if age >= 18:
                results.append({
                    'name': normalized_name,
                    'email': user.email,
                    'age': age
                })
    return results
```

## Principle: Use defaultdict to Avoid Key Checking

### Bad Practice: Manual Key Checking

```python
def group_users_by_country(users):
    groups = {}
    for user in users:
        country = user.country
        if country not in groups:
            groups[country] = []
        groups[country].append(user)
    return groups
```

**Why this is bad:**
- Repetitive key checking
- More verbose
- Easy to forget the check
- Not using Python's built-in tools

### Good Practice: Use defaultdict

```python
from collections import defaultdict

def group_users_by_country(users):
    groups = defaultdict(list)
    for user in users:
        groups[user.country].append(user)
    return dict(groups)  # Convert back if needed
```

**Why this is good:**
- No manual key checking needed
- More concise
- Less error-prone
- Leverages Python's standard library
- Intent is clearer
