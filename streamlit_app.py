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
    """Generate initial random population"""
    return [random.sample(range(num_programs), num_programs) for _ in range(pop_size)]

def fitness(individual, ratings):
    """Calculate total fitness for an individual"""
    return sum(ratings[i][pos] for pos, i in enumerate(individual))

def selection(population, fitnesses):
    """Select one individual using roulette-wheel selection"""
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(len(population), p=probabilities)]

def crossover(parent1, parent2):
    """Perform ordered crossover (OX)"""
    size = len(parent1)
    p1, p2 = random.sample(range(size), 2)
    start, end = min(p1, p2), max(p1, p2)
    child = [None] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child

def mutate(individual, mutation_rate):
    """Mutate an individual by swapping genes"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(ratings, crossover_rate, mutation_rate, seed, generations=100, pop_size=20):
    """Main GA routine"""
    random.seed(seed)
    np.random.seed(seed)
    num_programs = len(ratings)
    population = initialize_population(pop_size, num_programs)
    best_fitness_per_gen = []

    for _ in range(generations):
        fitnesses = [fitness(ind, ratings) for ind in population]
        ne
