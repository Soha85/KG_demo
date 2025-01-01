import spacy
import networkx as nx
import streamlit as st
from typing import List, Tuple, Dict
from spacy import displacy
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

class KnowledgeGraphBuilder:
    def __init__(self):
        """Initialize the knowledge graph builder with spaCy model."""
        self.nlp = spacy.load('en_core_web_md')
        self.graph = nx.DiGraph()
        self.doc = None
        self.pronoun_map = {}  # Dictionary to store pronoun -> noun mappings
        self.current_context = None  # Track current person being discussed
        
    def preprocess_text(self, text: str):
        """Preprocess the text and build pronoun mappings."""
        self.doc = self.nlp(text)
        self._build_pronoun_map()
        return self.doc
    
    def _build_pronoun_map(self):
        """Build a mapping of pronouns to their referent nouns with context awareness."""
        self.pronoun_map = {}
        self.current_context = None
        previous_pronoun = None
        
        for sent in self.doc.sents:
            for token in sent:
                # Reset context for each new subject
                if token.dep_ == "nsubj":
                    if token.pos_ == "PROPN":
                        self.current_context = token.text.lower()
                    elif token.pos_ == "PRON":
                        if previous_pronoun != token.text.lower():
                            self.current_context = None
                        
                    previous_pronoun = token.text.lower()
                    
                # Map pronouns to the current context
                if token.pos_ == "PRON" and self.current_context:
                    self.pronoun_map[token.i] = self.current_context
    
    def extract_entities(self, doc: spacy.tokens.Doc):
        """Extract entities and their types from the processed text."""
        entities = []
        seen = set()
    
        for token in doc:
            if token.pos_ in ["PROPN", "NOUN"] and token.dep_ in ["nsubj", "nsubjpass", "attr", "poss"]:
                entity_text = self._resolve_pronoun(token)
                entity_text = entity_text.lower()  # Normalize to lowercase
                if entity_text not in seen:
                    entities.append((entity_text, "NOUN"))
                    seen.add(entity_text)
    
        return entities
    
    def _resolve_pronoun(self, token):
        """Resolve a pronoun to its referent noun if possible."""
        if token.i in self.pronoun_map:
            return self.pronoun_map[token.i]
        return token.text.lower()
    
    def extract_relationships(self, doc: spacy.tokens.Doc):
        """Extract relationships with improved verb pattern handling."""
        relationships = []
    
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    subject = None
                    obj = None
                    verb_phrase = token.text
                
                    # Find subject
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = self._resolve_pronoun(child).lower()
                            break
                
                    # Find object
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr", "acomp"]:
                            obj = self._get_span_text(child).lower()
                            break
                
                    if subject and obj and subject != obj:
                        relationships.append((subject, verb_phrase, obj))
    
        return relationships
    
    def _get_span_text(self, token: spacy.tokens.Token):
        """Get the full text span for a token, including compound words and modifiers."""
        words = [token.text]
        
        # Check for compound words and adjective modifiers
        for child in token.children:
            if child.dep_ in ["compound", "amod"]:
                words.insert(0, child.text)
        
        return " ".join(words)
    
    def build_graph(self, text: str):
        """Build a knowledge graph from the input text."""
        doc = self.preprocess_text(text)
        entities = self.extract_entities(doc)
        relationships = self.extract_relationships(doc)
    
        self.graph.clear()
        seen_nodes = set()
        seen_edges = set()
    
        for entity, entity_type in entities:
            if entity not in seen_nodes:
                self.graph.add_node(entity, type=entity_type)
                seen_nodes.add(entity)
    
        for subj, pred, obj in relationships:
            if (subj, obj) not in seen_edges:
                self.graph.add_edge(subj, obj, relationship=pred)
                seen_edges.add((subj, obj))
    
        return self.graph
    
    # [Rest of the methods remain the same: get_dependency_viz, get_graph_info, visualize_graph]
    def get_dependency_viz(self) -> str:
        """Get HTML for dependency visualization."""
        if self.doc is None:
            return ""
        html = displacy.render(self.doc, style="dep", jupyter=False)
        return html
    
    def get_graph_info(self) -> Dict:
        """Return basic information about the knowledge graph."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
    
    def visualize_graph(self) -> plt.Figure:
        """Create a visualization of the knowledge graph."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=200, alpha=0.7)
        
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

    text = st.text_area("Enter your text here:", height=200,
                       placeholder="")
    
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
            if dep_html:
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
