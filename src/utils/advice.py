def generate_advice(user_input):
    """
    Generate loan approval improvement tips based on user input.

    Args:
        user_input (dict): Form input provided by user.

    Returns:
        list: List of advice strings.
    """
    advice = []

    # Check income
    if user_input.get('ApplicantIncome', 0) < 3000:
        advice.append("Consider increasing your monthly income to improve eligibility.")

    # Check loan amount
    if user_input.get('LoanAmount', 0) > 200:
        advice.append("Consider requesting a lower loan amount to enhance approval chances.")

    # Check credit history
    credit_history = user_input.get('Credit_History', '1.0')
    if credit_history == '0.0':
        advice.append("Building a positive credit history can significantly boost your chances.")

    return advice
