# Python Naming Conventions Best Practices

## Principle: Use Descriptive, Intention-Revealing Names

### Bad Practice: Cryptic Variable Names

```python
def process(d):
    r = []
    for i in d:
        if i['s'] == 'a':
            r.append(i['v'] * 1.1)
    return sum(r)
```

**Why this is bad:**
- Single-letter names give no context
- Requires reading the entire function to understand
- Hard to maintain and modify
- Easy to introduce bugs during refactoring

### Good Practice: Self-Documenting Names

```python
def calculate_total_active_subscriptions(subscriptions):
    active_values = []
    for subscription in subscriptions:
        if subscription['status'] == 'active':
            active_values.append(subscription['value'] * 1.1)
    return sum(active_values)
```

**Why this is good:**
- Function name describes what it does
- Variable names explain their purpose
- Code reads like a sentence
- Less need for comments

## Principle: Follow Python Naming Conventions (PEP 8)

### Bad Practice: Inconsistent Naming Styles

```python
class userAccount:  # Should be PascalCase
    def __init__(self, UserName, user_id):  # Inconsistent parameter naming
        self.UserName = UserName  # Should be snake_case
        self.userId = user_id  # Should be snake_case
        self.MAX_login_attempts = 3  # Mixing styles for constants

    def GetUserData(self):  # Should be snake_case
        return self.UserName
```

**Why this is bad:**
- Violates PEP 8 conventions
- Inconsistent style is confusing
- Harder to work with standard Python libraries
- Looks unprofessional

### Good Practice: Consistent PEP 8 Naming

```python
class UserAccount:
    MAX_LOGIN_ATTEMPTS = 3

    def __init__(self, username, user_id):
        self.username = username
        self.user_id = user_id
        self._login_attempts = 0

    def get_user_data(self):
        return self.username
```

**Why this is good:**
- Class name in PascalCase
- Methods and variables in snake_case
- Constants in UPPER_SNAKE_CASE
- Private attributes prefixed with underscore
- Follows Python community standards

## Principle: Avoid Meaningless Names

### Bad Practice: Non-Descriptive Names

```python
def do_stuff(data1, data2):
    temp = data1
    temp2 = data2
    result = temp + temp2
    final = result * 2
    return final
```

**Why this is bad:**
- "stuff", "temp", "final" convey no meaning
- Numbered variables suggest poor design
- No indication of what the function does
- Hard to understand the business logic

### Good Practice: Meaningful Domain Names

```python
def calculate_compound_interest(principal, interest_rate):
    base_amount = principal
    interest = principal * interest_rate
    total_with_interest = base_amount + interest
    return total_with_interest
```

**Why this is good:**
- Names describe financial concepts
- Clear what each variable represents
- Function name describes the calculation
- Domain language makes code understandable to non-programmers
