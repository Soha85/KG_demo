import spacy
import networkx as nx
import streamlit as st
from typing import List, Tuple, Dict
import en_core_web_md
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self):
        """Initialize the knowledge graph builder with spaCy medium model."""
        self.nlp = spacy.load('en_core_web_md')
        self.graph = nx.DiGraph()
        
    def preprocess_text(self, text: str):
        """Preprocess the input text using spaCy."""
        return self.nlp(text)
    
    def extract_entities(self, doc: spacy.tokens.Doc):
        """Extract entities and their types from the processed text."""
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        return entities
    
    def extract_relationships(self, doc: spacy.tokens.Doc):
        """Extract relationships between entities using dependency parsing."""
        relationships = []
        
        for token in doc:
            # Look for subject-verb-object patterns
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child
                        break
                
                # Find object
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        obj = child
                        break
                
                if subject and obj:
                    relationships.append((
                        self._get_span_text(subject),
                        token.text,
                        self._get_span_text(obj)
                    ))
        
        return relationships
    
    def _get_span_text(self, token: spacy.tokens.Token):
        """Get the full text span for a token, including compound words and modifiers."""
        words = [token.text]
        
        # Check for compound words
        for child in token.children:
            if child.dep_ == "compound":
                words.insert(0, child.text)
        
        return " ".join(words)
    
    def build_graph(self, text: str):
        """Build a knowledge graph from the input text."""
        # Process text
        doc = self.preprocess_text(text)
        
        # Extract entities and relationships
        entities = self.extract_entities(doc)
        relationships = self.extract_relationships(doc)
        
        # Clear previous graph
        self.graph.clear()
        
        # Add entities to graph
        for entity, entity_type in entities:
            self.graph.add_node(entity, type=entity_type)
        
        # Add relationships to graph
        for subj, pred, obj in relationships:
            self.graph.add_edge(subj, obj, relationship=pred)
        
        return self.graph
    
    def get_graph_info(self):
        """Return basic information about the knowledge graph."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
    
    def visualize_graph(self):
        """Create a visualization of the knowledge graph."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                             arrows=True, arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(self.graph, pos)
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'relationship')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        return plt.gcf()

def main():
    st.title("Knowledge Graph Builder")
    
  
    # Create text input area
    text = st.text_area("Enter your text here:", height=200,
                       placeholder="Enter text to analyze... (e.g., 'Albert Einstein developed the theory of relativity.')")
    
    # Initialize knowledge graph builder
    kg_builder = KnowledgeGraphBuilder()
    
    if st.button("Generate Knowledge Graph"):
        if text.strip():
            # Build graph
            graph = kg_builder.build_graph(text)
            graph_info = kg_builder.get_graph_info()
            
            # Display basic information
            st.subheader("Graph Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Nodes", graph_info['num_nodes'])
            with col2:
                st.metric("Number of Edges", graph_info['num_edges'])
            
            # Display entities
            st.subheader("Entities Found")
            for node, attrs in graph_info['nodes']:
                st.write(f"- {node} ({attrs.get('type', 'Unknown type')})")
            
            # Display relationships
            st.subheader("Relationships")
            for subj, obj, attrs in graph_info['edges']:
                st.write(f"- {subj} --[{attrs['relationship']}]--> {obj}")
            
            # Visualize graph
            st.subheader("Graph Visualization")
            fig = kg_builder.visualize_graph()
            st.pyplot(fig)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
