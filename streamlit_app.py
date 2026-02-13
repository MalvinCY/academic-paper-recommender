"""
Academic Paper Recommender - Streamlit Interface

Discover relevant research papers using SPECTER2 embeddings and FAISS similarity search.
"""

import streamlit as st
import numpy as np
import pandas as pd
import faiss
import os

# Page config
st.set_page_config(
    page_title="Academic Paper Recommender", page_icon="📚", layout="wide"
)

# Paths
DATA_DIR = "data/processed"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_normalized.npy")
INDEX_PATH = os.path.join(DATA_DIR, "papers.index")
PAPERS_PATH = os.path.join(DATA_DIR, "papers_with_embeddings.pkl")


# Category formatting function
def format_category(cat):
    """Convert arXiv category codes to readable names"""
    category_names = {
        # Computer Science
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation & Language (NLP)",
        "cs.CR": "Cryptography & Security",
        "cs.CV": "Computer Vision",
        "cs.CY": "Computers & Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures & Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science & Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural & Evolutionary Computing",
        "cs.NI": "Networking & Internet",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social & Information Networks",
        "cs.SY": "Systems & Control",
        # Statistics
        "stat.AP": "Statistics - Applications",
        "stat.CO": "Statistics - Computation",
        "stat.ME": "Statistics - Methodology",
        "stat.ML": "Statistics - Machine Learning",
        "stat.OT": "Statistics - Other",
        "stat.TH": "Statistics - Theory",
        # Mathematics
        "math.AC": "Commutative Algebra",
        "math.AG": "Algebraic Geometry",
        "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology",
        "math.CA": "Classical Analysis",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics",
        "math.GN": "General Topology",
        "math.GR": "Group Theory",
        "math.GT": "Geometric Topology",
        "math.HO": "History & Overview",
        "math.IT": "Information Theory",
        "math.KT": "K-Theory",
        "math.LO": "Logic",
        "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics",
        "math.NA": "Numerical Analysis",
        "math.NT": "Number Theory",
        "math.OA": "Operator Algebras",
        "math.OC": "Optimization & Control",
        "math.PR": "Probability",
        "math.QA": "Quantum Algebra",
        "math.RA": "Rings & Algebras",
        "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry",
        "math.SP": "Spectral Theory",
        "math.ST": "Statistics Theory",
        # Physics
        "astro-ph": "Astrophysics",
        "astro-ph.CO": "Cosmology & Nongalactic",
        "astro-ph.EP": "Earth & Planetary",
        "astro-ph.GA": "Astrophysics of Galaxies",
        "astro-ph.HE": "High Energy Astrophysics",
        "astro-ph.IM": "Instrumentation & Methods",
        "astro-ph.SR": "Solar & Stellar",
        "cond-mat.dis-nn": "Disordered Systems",
        "cond-mat.mes-hall": "Mesoscale & Nanoscale",
        "cond-mat.mtrl-sci": "Materials Science",
        "cond-mat.other": "Other Condensed Matter",
        "cond-mat.quant-gas": "Quantum Gases",
        "cond-mat.soft": "Soft Condensed Matter",
        "cond-mat.stat-mech": "Statistical Mechanics",
        "cond-mat.str-el": "Strongly Correlated",
        "cond-mat.supr-con": "Superconductivity",
        "gr-qc": "General Relativity",
        "hep-ex": "High Energy Physics - Experiment",
        "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology",
        "hep-th": "High Energy Physics - Theory",
        "math-ph": "Mathematical Physics",
        "nlin.AO": "Adaptation & Self-Organizing",
        "nlin.CD": "Chaotic Dynamics",
        "nlin.CG": "Cellular Automata",
        "nlin.PS": "Pattern Formation",
        "nlin.SI": "Exactly Solvable",
        "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory",
        "physics.acc-ph": "Accelerator Physics",
        "physics.ao-ph": "Atmospheric & Oceanic",
        "physics.app-ph": "Applied Physics",
        "physics.atm-clus": "Atomic & Molecular Clusters",
        "physics.atom-ph": "Atomic Physics",
        "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics",
        "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics",
        "physics.data-an": "Data Analysis",
        "physics.ed-ph": "Physics Education",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics",
        "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History & Philosophy",
        "physics.ins-det": "Instrumentation & Detectors",
        "physics.med-ph": "Medical Physics",
        "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics",
        "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Physics & Society",
        "physics.space-ph": "Space Physics",
        "quant-ph": "Quantum Physics",
        # Other
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons & Cognition",
        "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations & Evolution",
        "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes",
        "q-bio.TO": "Tissues & Organs",
        "q-fin.CP": "Computational Finance",
        "q-fin.EC": "Economics",
        "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance",
        "q-fin.PM": "Portfolio Management",
        "q-fin.PR": "Pricing of Securities",
        "q-fin.RM": "Risk Management",
        "q-fin.ST": "Statistical Finance",
        "q-fin.TR": "Trading & Market",
        "econ.EM": "Econometrics",
        "econ.GN": "General Economics",
        "econ.TH": "Theoretical Economics",
        "eess.AS": "Audio & Speech Processing",
        "eess.IV": "Image & Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems & Control",
    }
    return category_names.get(cat, cat)


def get_main_discipline(cat):
    """Extract main discipline from category code"""
    if cat.startswith("cs."):
        return "Computer Science"
    elif cat.startswith("stat."):
        return "Statistics"
    elif cat.startswith("math."):
        return "Mathematics"
    elif cat.startswith("physics.") or cat in [
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math-ph",
        "nucl-ex",
        "nucl-th",
        "quant-ph",
    ]:
        return "Physics"
    elif cat.startswith("astro-ph"):
        return "Astrophysics"
    elif cat.startswith("cond-mat"):
        return "Condensed Matter"
    elif cat.startswith("nlin."):
        return "Nonlinear Sciences"
    elif cat.startswith("q-bio."):
        return "Quantitative Biology"
    elif cat.startswith("q-fin."):
        return "Quantitative Finance"
    elif cat.startswith("econ."):
        return "Economics"
    elif cat.startswith("eess."):
        return "Electrical Engineering"
    else:
        return "Other"


# Cache data loading
@st.cache_resource
def load_data():
    """Load embeddings, FAISS index, and paper metadata"""
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_pickle(PAPERS_PATH)
    return embeddings, index, df


# Load data
with st.spinner("Loading data..."):
    embeddings, index, df = load_data()


# Helper function
def get_recommendations(paper_idx, k=5):
    """Get k similar papers"""
    query_vector = embeddings[paper_idx : paper_idx + 1].astype("float32")
    distances, indices = index.search(query_vector, k + 1)

    # Skip first result (query itself)
    result_indices = indices[0][1:]
    result_distances = distances[0][1:]

    # Convert to similarities
    similarities = 1 - (result_distances**2) / 2

    return result_indices, similarities


# ============= UI =============

st.title("📚 Academic Paper Recommender")
st.markdown("*Discover relevant research papers using semantic similarity*")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you discover relevant academic papers using:
    - **SPECTER2** embeddings (trained on citation relationships)
    - **FAISS** for fast similarity search
    - **9,280** papers from arXiv (2008-2026)
    """)

    st.header("Dataset Statistics")
    st.metric("Total Papers", f"{len(df):,}")
    st.metric("Categories", df["primary_category"].nunique())
    st.metric("Date Range", f"{df['year'].min()}-{df['year'].max()}")

    st.header("How to Use")
    st.markdown("""
    1. **Find Similar Papers**: Enter a paper ID or search by title
    2. **Explore Random**: Discover papers randomly
    3. **Browse by Category**: Filter by discipline and category
    
    💡 **Tip**: Click "Get Similar Papers" on any paper to find related research!
    """)

# Main content
tab1, tab2, tab3 = st.tabs(
    ["🔍 Find Similar Papers", "🎲 Explore Random", "📊 Browse by Category"]
)

# Tab 1: Find Similar Papers
with tab1:
    st.header("Find Papers Similar to...")

    # Check if paper_id came from URL parameter
    query_params = st.query_params
    url_paper_id = query_params.get("paper_id", "")

    # Search by paper ID or title
    search_method = st.radio(
        "Search by:", ["Paper ID", "Title Keywords"], horizontal=True
    )

    if search_method == "Paper ID":
        paper_id_input = st.text_input(
            "Enter arXiv Paper ID:",
            value=url_paper_id,  # Pre-fill if from URL
            placeholder="e.g., 2301.07041 or 1706.03762v1",
            help="Try the 'Explore Random' tab to get valid paper IDs",
        )

        # Auto-trigger if URL has paper_id
        should_search = bool(url_paper_id) or st.button(
            "Get Recommendations", type="primary"
        )

        if paper_id_input and should_search:
            matching = df[df["paper_id"] == paper_id_input]

            if len(matching) == 0:
                st.error(
                    f"Paper '{paper_id_input}' not found. Try searching by title or use 'Explore Random'."
                )
            else:
                query_paper = matching.iloc[0]
                query_idx = matching.index[0]

                # Display query paper
                st.subheader("📄 Query Paper")
                with st.container():
                    st.markdown(f"**{query_paper['title']}**")
                    st.caption(
                        f"Categories: {', '.join(eval(query_paper['categories']))} | Published: {query_paper['published']}"
                    )
                    with st.expander("View Abstract"):
                        st.write(query_paper["abstract"])

                # Get recommendations
                num_recs = st.slider("Number of recommendations:", 3, 10, 5)
                rec_indices, similarities = get_recommendations(query_idx, k=num_recs)

                st.subheader(f"🎯 Top {num_recs} Similar Papers")

                for i, (idx, sim) in enumerate(zip(rec_indices, similarities), 1):
                    paper = df.iloc[idx]

                    with st.container():
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.markdown(f"**{i}. {paper['title']}**")
                            st.caption(
                                f"Categories: {', '.join(eval(paper['categories']))} | Published: {paper['published']}"
                            )
                            with st.expander("View Abstract"):
                                st.write(paper["abstract"])

                        with col2:
                            st.metric("Similarity", f"{sim:.3f}")
                            st.link_button(
                                "PDF", paper["pdf_url"], use_container_width=True
                            )

                        st.divider()

    else:  # Title Keywords
        title_query = st.text_input(
            "Search for keywords in titles:",
            placeholder="e.g., transformer, reinforcement learning, computer vision",
        )

        if title_query:
            matches = df[df["title"].str.contains(title_query, case=False, na=False)]

            if len(matches) == 0:
                st.warning(f"No papers found with '{title_query}' in the title.")
            else:
                st.success(f"Found {len(matches)} papers matching '{title_query}'")

                # Show first 10 matches
                for _, paper in matches.head(10).iterrows():
                    with st.expander(f"📄 {paper['title']}"):
                        st.caption(
                            f"ID: {paper['paper_id']} | Categories: {', '.join(eval(paper['categories']))}"
                        )
                        st.write(paper["abstract"][:300] + "...")

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.link_button(
                                "🔍 Get Similar Papers",
                                f"?paper_id={paper['paper_id']}",
                                use_container_width=True,
                            )
                        with col2:
                            st.link_button(
                                "📄 PDF", paper["pdf_url"], use_container_width=True
                            )

# Tab 2: Explore Random
with tab2:
    st.header("🎲 Discover Random Papers")

    if st.button("Show Me a Random Paper", type="primary"):
        random_paper = df.sample(1).iloc[0]
        random_idx = df[df["paper_id"] == random_paper["paper_id"]].index[0]

        # Display random paper
        st.subheader("📄 Random Paper")
        with st.container():
            st.markdown(f"**{random_paper['title']}**")
            st.caption(
                f"ID: {random_paper['paper_id']} | Categories: {', '.join(eval(random_paper['categories']))} | Published: {random_paper['published']}"
            )
            with st.expander("View Abstract"):
                st.write(random_paper["abstract"])

            col1, col2 = st.columns(2)
            with col1:
                st.link_button(
                    "🔍 Get Similar Papers",
                    f"?paper_id={random_paper['paper_id']}",
                    use_container_width=True,
                )
            with col2:
                st.link_button(
                    "📄 View PDF", random_paper["pdf_url"], use_container_width=True
                )

        # Show similar papers
        st.subheader("🎯 Similar Papers")
        rec_indices, similarities = get_recommendations(random_idx, k=5)

        for i, (idx, sim) in enumerate(zip(rec_indices, similarities), 1):
            paper = df.iloc[idx]

            with st.expander(f"{i}. {paper['title']} (Similarity: {sim:.3f})"):
                st.caption(
                    f"Categories: {', '.join(eval(paper['categories']))} | Published: {paper['published']}"
                )
                st.write(paper["abstract"][:300] + "...")
                col1, col2 = st.columns(2)
                with col1:
                    st.link_button(
                        "🔍 Get Similar Papers",
                        f"?paper_id={paper['paper_id']}",
                        use_container_width=True,
                    )
                with col2:
                    st.link_button(
                        "📄 View PDF", paper["pdf_url"], use_container_width=True
                    )

# Tab 3: Browse by Category
with tab3:
    st.header("📊 Browse Papers by Category")

    # Organize categories by discipline
    categories_by_discipline = {}
    for cat in df["primary_category"].unique():
        discipline = get_main_discipline(cat)
        if discipline not in categories_by_discipline:
            categories_by_discipline[discipline] = []
        categories_by_discipline[discipline].append(cat)

    # Sort disciplines and their categories
    for discipline in categories_by_discipline:
        categories_by_discipline[discipline].sort()

    # First dropdown: Select discipline
    col1, col2 = st.columns(2)

    with col1:
        disciplines = sorted(categories_by_discipline.keys())
        selected_discipline = st.selectbox(
            "1. Select Main Category:",
            disciplines,
            index=disciplines.index("Computer Science")
            if "Computer Science" in disciplines
            else 0,
        )

    with col2:
        # Second dropdown: Select specific category within discipline
        available_categories = categories_by_discipline[selected_discipline]
        category_options = {
            f"{format_category(cat)}": cat for cat in available_categories
        }

        selected_display = st.selectbox(
            "2. Select Specific Category:", list(category_options.keys())
        )
        selected_cat = category_options[selected_display]

    # Show papers in selected category
    cat_papers = df[df["primary_category"] == selected_cat].copy()

    # Info box with stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers in Category", f"{len(cat_papers):,}")
    with col2:
        if len(cat_papers) > 0:
            st.metric(
                "Date Range", f"{cat_papers['year'].min()}-{cat_papers['year'].max()}"
            )
    with col3:
        if len(cat_papers) > 0:
            st.metric(
                "Avg Abstract Length",
                f"{int(cat_papers['abstract_length'].mean())} chars",
            )

    st.divider()

    # Sort options
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.radio("Sort by:", ["Most Recent", "Oldest First"], horizontal=True)
    with col2:
        papers_per_page = st.selectbox("Papers per page:", [10, 20, 50], index=1)

    # Sort papers
    if sort_by == "Most Recent":
        cat_papers = cat_papers.sort_values("published", ascending=False)
    else:  # Oldest First
        cat_papers = cat_papers.sort_values("published", ascending=True)

    # Pagination
    total_papers = len(cat_papers)
    total_pages = (total_papers + papers_per_page - 1) // papers_per_page

    # Initialize page number in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Reset page when category changes
    if (
        "last_category" not in st.session_state
        or st.session_state.last_category != selected_cat
    ):
        st.session_state.current_page = 1
        st.session_state.last_category = selected_cat

    # Calculate start and end indices
    start_idx = (st.session_state.current_page - 1) * papers_per_page
    end_idx = min(start_idx + papers_per_page, total_papers)

    # Display current page papers
    current_page_papers = cat_papers.iloc[start_idx:end_idx]

    for i, (_, paper) in enumerate(current_page_papers.iterrows(), start_idx + 1):
        with st.expander(f"📄 {i}. {paper['title']} ({paper['published'][:4]})"):
            st.caption(
                f"**ID:** {paper['paper_id']} | **Published:** {paper['published']}"
            )
            st.write(paper["abstract"][:400] + "...")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.link_button(
                    "🔍 Get Similar Papers",
                    f"?paper_id={paper['paper_id']}",
                    use_container_width=True,
                )
            with col2:
                st.link_button("📄 PDF", paper["pdf_url"], use_container_width=True)

    # Pagination controls
    if total_pages > 1:
        st.divider()

        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

        with col1:
            if st.button("⏮️ First", disabled=(st.session_state.current_page == 1)):
                st.session_state.current_page = 1
                st.rerun()

        with col2:
            if st.button("◀️ Prev", disabled=(st.session_state.current_page == 1)):
                st.session_state.current_page -= 1
                st.rerun()

        with col3:
            st.markdown(
                f"<div style='text-align: center; padding-top: 5px;'>Page {st.session_state.current_page} of {total_pages} ({total_papers} total papers)</div>",
                unsafe_allow_html=True,
            )

        with col4:
            if st.button(
                "Next ▶️", disabled=(st.session_state.current_page == total_pages)
            ):
                st.session_state.current_page += 1
                st.rerun()

        with col5:
            if st.button(
                "Last ⏭️", disabled=(st.session_state.current_page == total_pages)
            ):
                st.session_state.current_page = total_pages
                st.rerun()

# Footer
st.divider()
st.caption(
    "Built with SPECTER2 embeddings, FAISS similarity search, and Streamlit | Data from arXiv API"
)
