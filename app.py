import spacy
import networkx as nx
import streamlit as st
from typing import List, Tuple, Dict
import en_core_web_md
from collections import defaultdict
import matplotlib.pyplot as plt
from spacy import displacy
import streamlit.components.v1 as components

class KnowledgeGraphBuilder:
    # [Previous class code remains the same until build_graph method]
    
    def build_graph(self, text: str) -> nx.DiGraph:
        """Build a knowledge graph from the input text."""
        # Process text
        self.doc = self.preprocess_text(text)  # Store doc as instance variable
        
        # Extract entities and relationships
        entities = self.extract_entities(self.doc)
        relationships = self.extract_relationships(self.doc)
        
        # Clear previous graph
        self.graph.clear()
        
        # Add entities to graph
        for entity, entity_type in entities:
            self.graph.add_node(entity, type=entity_type)
        
        # Add relationships to graph
        for subj, pred, obj in relationships:
            self.graph.add_edge(subj, obj, relationship=pred)
        
        return self.graph
    
    def get_dependency_viz(self) -> str:
        """Get HTML for dependency visualization."""
        html = displacy.render(self.doc, style="dep", jupyter=False)
        return html

def main():
    st.title("Knowledge Graph Builder")
    
    # Add description
    st.write("""
    This application builds a knowledge graph from input text using Natural Language Processing.
    Enter your text below to visualize the relationships between entities.
    """)
    
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
            
            # Display dependency visualization
            st.subheader("Dependency Parse")
            dep_html = kg_builder.get_dependency_viz()
            components.html(dep_html, height=400, scrolling=True)
            
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
