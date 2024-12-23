import spacy
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("Interactive Knowledge Graph")

# Step 1: Extract Entities and Relations
def extract_entities_and_relations(doc):
    entities = []
    relations = []
    
    # Extract named entities
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    # Extract relations using dependency parsing
    for token in doc:
        # Look for meaningful relations
        if token.dep_ in ("nsubj", "dobj"):# and token.head.ent_type_:
            head = token.head.text
            tail = token.text
            relation = token.dep_
            relations.append((head, relation, tail))
    
    return entities, relations

# Input Text Area
document = st.text_area("Input Text:", 
    """
    Albert Einstein developed the Theory of Relativity. He was born in Germany. 
    Einstein won the Nobel Prize in Physics in 1921. The capital of France is Paris.
    """)

if st.button('Generate Knowledge Graph'):
    # Process the document
    doc = nlp(document)
    entities, relations = extract_entities_and_relations(doc)

    # Display extracted entities and relations
    st.write("### Extracted Entities:")
    st.write(entities)
    st.write("### Extracted Relations:")
    st.write(relations)

    # Step 2: Build Knowledge Graph
    G = nx.DiGraph()

    # Add entities as nodes
    for entity, label in entities:
        G.add_node(entity, type=label)

    # Add relations as edges
    for head, relation, tail in set(relations):  # Avoid duplicate relations
        if head in G.nodes and tail in G.nodes:
            G.add_edge(head, tail, relation=relation)

    # Add missing nodes dynamically
    for head, _, tail in relations:
        if head not in G.nodes:
            G.add_node(head)
        if tail not in G.nodes:
            G.add_node(tail)

    # Step 3: Visualize the Knowledge Graph
    st.write("### Knowledge Graph Visualization")
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)

    # Draw nodes with color based on entity type
    node_colors = [
        "lightblue" if G.nodes[node].get("type") == "PERSON" else "orange"
        for node in G.nodes
    ]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="black")

    # Add labels to nodes and edges
    node_labels = {node: f"{node}\n({G.nodes[node].get('type', 'N/A')})" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight="bold")

    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    st.write(G)
    #plt.title("Knowledge Graph from Document", fontsize=12)
    plt.axis("off")

    # Render the graph in Streamlit
    st.pyplot(plt)
