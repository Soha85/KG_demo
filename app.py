import spacy
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("Sample Knowledge Graph:")


# Step 1: Extract Entities and Relations
def extract_entities_and_relations(doc):
    entities = []
    relations = []

    for sent in doc.sents:
        # Extract named entities
        sent_entities = [(ent.text, ent.label_) for ent in sent.ents]
        entities.extend(sent_entities)

        # Dummy relation extraction using dependency parsing
        for token in sent:
            if token.dep_ in ("ROOT", "attr", "agent", "dobj"):
                head = token.head.text
                tail = token.text
                relations.append((head, token.dep_, tail))

    return entities, relations

document = st.text_area("Input Data:")
if st.button('Apply Knowledge Graph'):


    # Process the document
    doc = nlp(document)
    entities, relations = extract_entities_and_relations(doc)

    # Step 2: Build Knowledge Graph
    G = nx.DiGraph()

    # Add entities as nodes
    for entity, label in entities:
        G.add_node(entity, label=label)

    # Add relations as edges
    for head, relation, tail in relations:
        if head in G.nodes and tail in G.nodes:
            G.add_edge(head, tail, relation=relation)

    # Step 3: Visualize the Graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", alpha=0.9)

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Knowledge Graph from Document", fontsize=14)
    plt.axis("off")
    plt.show()