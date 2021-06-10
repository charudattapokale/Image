# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 01:38:49 2021

@author: charu
"""

import streamlit as st
import os

filename = st.text_input('Enter a file path:')
try:
    with open(filename) as input:
        st.text(input.read())
except FileNotFoundError:
    st.error('File not found.')