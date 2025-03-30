import streamlit as st

st.title("Contact Us")
    
# Introduction
st.markdown("""
## We're Here to Help

Have a question about our loan simulation tools? Need assistance with your financial planning? 
Our team is ready to assist you. Fill out the form below, and we'll get back to you via email 
as soon as possible.
""")

# Explanation of approach
st.info("""
**Our Commitment to You:**

We strive to respond to all inquiries within 1-2 business days. Your questions are important to us, 
and we're dedicated to providing thorough, personalized responses via email. This allows us to:

- Research your specific situation properly
- Provide detailed, accurate information
- Include relevant resources and documentation
- Ensure you have a written record of our advice for future reference
""")

# Contact form
with st.form("contact_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name*")
        email = st.text_input("Email Address*")
    
    with col2:
        phone = st.text_input("Phone Number (optional)")
        subject = st.text_input("Subject*")
    
    # Question/message box
    message = st.text_area("Your Question or Message*", height=150)
    
    # Checkbox for terms
    agree = st.checkbox("I agree to receive email responses to my inquiry")
    
    # Submit button
    submitted = st.form_submit_button("Submit Question")
    
    if submitted:
        
        if name and email and subject and message and agree:
            # Here you would typically implement email sending functionality
            # For now, we'll just show a success message
            st.success("Thank you for your message! We've received your inquiry and will respond to your email shortly.")
            
            # You could log the message to a database or send via email service
            # For demonstration, we'll just print to console
            st.write("Message details (this would be sent to your backend):")
            st.json({
                "name": name,
                "email": email,
                "phone": phone,
                "subject": subject,
                "message": message
            })
        else:
            st.error("Please fill out all required fields and accept the terms to submit your question.")

# Additional contact information
st.markdown("""
## Additional Ways to Reach Us

**Business Hours:**  
Monday - Friday: 9:00 AM - 5:00 PM

**Phone:**  
(555) 123-4567

**Email:**  
support@loansimulator.example.com

**Address:**  
123 Financial Street  
Suite 400  
New York, NY 10001
""")