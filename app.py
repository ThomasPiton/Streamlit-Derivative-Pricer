import streamlit as st
from streamlit_option_menu import option_menu
# from config import *

# st.logo(
#     "static/img/bank_logo.png",
#     icon_image="static/img/bank_logo.png")

pages = {
    "Pricer": [
        st.Page(page="pages/display_pricer/page1.py", title="Page-1", icon=":material/credit_score:"),
        st.Page(page="pages/display_pricer/page2.py", title="Page-2", icon=":material/credit_score:"),
    ],
    "Others": [
        st.Page("pages/others/contact.py", title="Contact", icon=":material/contacts_product:"),
        st.Page("pages/others/help.py", title="Need help ?", icon=":material/help:"),
    ],
}

pg = st.navigation(pages)
pg.run()