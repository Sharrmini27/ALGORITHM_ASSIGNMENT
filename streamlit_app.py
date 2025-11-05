import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# STREAMLIT PAGE SETTINGS
# ==========================================
st.set_page_config(page_title="AI-Powered TV Program Scheduler", layout="wide")
st.title("📺 AI-Powered Genetic Algorithm for TV Program Scheduling")
st.caption("Developed by **Sharrmini Veeran** — Intelligent Scheduling Optimization System")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("program_ratings (2).csv")
    return df

df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# ==========================================
# GENETIC ALGORITHM TRIAL CONFIGURATION
# ==========================================
trials = [
    {"trial": "Trial 1", "crossover": 0.8, "mutation": 0.2, "seed": 11},
    {"trial": "Trial 2", "crossover": 0.9, "mutation": 0.1, "seed": 22},
    {"trial": "Trial 3", "crossover": 0.7, "mutation": 0.3, "seed": 33},
]

# ==========================================
# GENETIC ALGORITHM FUNCTIONS
# ==========================================
def initialize_population(pop_size, num_programs):
    return [random.sample(range(num_programs), num_programs) for _ in range(pop_size)]

def fitness(individual, ratings):
    return sum(ratings[i][pos] for pos, i in enumerate(individual))

def selection(population, fitnesses):
    total_fitness = sum(fitnesses_
