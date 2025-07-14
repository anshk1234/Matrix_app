import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from streamlit_lottie import st_lottie

# --- Splash Animation ---
def load_lottiefile(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

if st.session_state.show_intro:
    lottie_intro = load_lottiefile("Calculator.json")
    splash = st.empty()
    with splash.container():
        st.markdown("<h1 style='text-align:center;'>Welcome to MATRIX CALCULATOR!</h1>", unsafe_allow_html=True)
        st_lottie(lottie_intro, height=280, speed=1.0, loop=False)
        time.sleep(4)
    splash.empty()
    st.session_state.show_intro = False

# Title
st.title("🔢 Interactive Matrix Calculator")

# Step 1: Matrix A (Main Matrix)
st.sidebar.header("Matrix A Settings")
rows_A = st.sidebar.number_input("Rows for Matrix A:", min_value=2, max_value=5, value=3)
cols_A = st.sidebar.number_input("Columns for Matrix A:", min_value=2, max_value=5, value=3)

st.write("### ✏️ Enter Elements for Matrix A:")
matrix_A_data = pd.DataFrame(np.zeros((rows_A, cols_A)), columns=[f"Col {i+1}" for i in range(cols_A)])
edited_matrix_A = st.data_editor(matrix_A_data, key="matrixA_editor")
matrix_A = edited_matrix_A.to_numpy()

st.write("✅ Your Updated Matrix A:")
st.write(matrix_A)

# Step 2: Matrix A Operations
st.subheader("📊 Operations on Matrix A")

# Transpose
transpose = np.transpose(matrix_A)
st.subheader("🔹 Transpose of Matrix A:")
st.write(transpose)

# Determinant, Adjoint, Inverse (for square matrices)
if rows_A == cols_A:
    det = np.linalg.det(matrix_A)
    st.subheader("🔹 Determinant of Matrix A:")
    st.write(f"{det:.0f}")

    def cofactor_matrix(matrix):
        cofactors = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                cofactors[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        return cofactors

    adjoint = cofactor_matrix(matrix_A).T
    st.subheader("🔹 Adjoint (Adjugate) of Matrix A:")
    st.write(adjoint)

    if det == 0:
        st.warning("⚠️ Matrix A is singular! Inverse does not exist.")
    else:
        inverse = np.linalg.inv(matrix_A)
        st.subheader("🔹 Inverse of Matrix A:")
        st.write(inverse)
else:
    st.warning("⚠️ Matrix A must be square for determinant, adjoint, and inverse.")

# Step 3: Matrix B (For Multiplication)
st.sidebar.header("Matrix B Settings")
rows_B = st.sidebar.number_input("Rows for Matrix B:", min_value=2, max_value=5, value=cols_A)
cols_B = st.sidebar.number_input("Columns for Matrix B:", min_value=2, max_value=5, value=3)

st.write("### ✏️ Enter Elements for Matrix B:")
matrix_B_data = pd.DataFrame(np.zeros((rows_B, cols_B)), columns=[f"Col {i+1}" for i in range(cols_B)])
edited_matrix_B = st.data_editor(matrix_B_data, key="matrixB_editor")
matrix_B = edited_matrix_B.to_numpy()

# Step 4: Multiplication
st.subheader("🔹 Matrix Multiplication A × B")
if matrix_A.shape[1] == matrix_B.shape[0]:
    result = np.dot(matrix_A, matrix_B)
    st.write(result)
else:
    st.warning("⚠️ Cannot multiply! Columns of Matrix A must match rows of Matrix B.")

st.success("🎯 Matrix Calculations Complete!")
