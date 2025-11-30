def find_duplicate_users(user_ids: list[str]) -> list[str]:
    seen: list[str] = []
    duplicates: list[str] = []

    for user_id in user_ids:
        if user_id in seen:
            duplicates.append(user_id)
        else:
            seen.append(user_id)

    return duplicates
