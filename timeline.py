import pandas as pd
import plotly.express as px
from SPARQLWrapper import SPARQLWrapper, JSON
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Historical Events Timeline",
    page_icon="📅",
    layout="wide"
)

@st.cache_data(ttl=3600)
def get_timeline_data(start_year, end_year, limit, event_types):
    """Queries DBPedia for historical events and returns a DataFrame."""
    
    endpoint_url = "https://dbpedia.org/sparql"
    
    # Build query blocks based on selected event types
    query_blocks = []
    
    if "Wars" in event_types:
        query_blocks.append("""
        {
          SELECT ?label ?startDate ?endDate ("War" as ?type)
          WHERE {
            ?event a dbo:MilitaryConflict ;
                   rdfs:label ?label ;
                   dbo:startDate ?startDate ;
                   dbo:endDate ?endDate .
            FILTER (lang(?label) = "en")
          }
        }
        """)
    
    if "Births" in event_types:
        query_blocks.append("""
        {
          SELECT ?label ?startDate ?endDate ("Birth" as ?type)
          WHERE {
            ?event a dbo:Person ;
                   rdfs:label ?label ;
                   dbo:birthDate ?startDate .
            BIND(?startDate as ?endDate) 
            FILTER (lang(?label) = "en")
          }
        }
        """)
    
    if "Organizations" in event_types:
        query_blocks.append("""
        {
          SELECT ?label ?startDate ?endDate ("Founding" as ?type)
          WHERE {
            ?event a dbo:Organisation ;
                   rdfs:label ?label ;
                   dbo:formationDate ?startDate .
            BIND(?startDate as ?endDate)
            FILTER (lang(?label) = "en")
          }
        }
        """)
    
    if not query_blocks:
        return pd.DataFrame()
    
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?label ?startDate ?endDate ?type
    WHERE {{
      FILTER(?startDate >= "{start_year}-01-01"^^xsd:date && ?startDate < "{end_year}-01-01"^^xsd:date)
      
      {" UNION ".join(query_blocks)}
    }}
    ORDER BY ?startDate
    LIMIT {limit}
    """
    
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    processed_results = []
    for result in results["results"]["bindings"]:
        processed_results.append({
            'Task': result['label']['value'],
            'Start': result['startDate']['value'],
            'Finish': result['endDate']['value'],
            'Type': result['type']['value']
        })
    
    if not processed_results:
        return pd.DataFrame()

    df = pd.DataFrame(processed_results)
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Finish'] = pd.to_datetime(df['Finish'], errors='coerce')
    df = df.dropna(subset=['Start'])
    df['Finish'] = df['Finish'].fillna(df['Start'])
    
    # Fix for single-day events
    mask = (df['Finish'] == df['Start'])
    df.loc[mask, 'Finish'] = df.loc[mask, 'Start'] + pd.DateOffset(days=1)
    
    return df

def create_timeline(df):
    """Creates and returns an interactive timeline chart."""
    
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Task",
        color="Type",
        hover_name="Task",
        hover_data={
            "Start": "|%Y-%m-%d",
            "Finish": "|%Y-%m-%d",
            "Type": True,
            "Task": False
        },
        title="Historical Events Timeline"
    )

    fig.update_yaxes(autorange="reversed") 
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="",
        height=800,
        title_x=0.5,
        yaxis=dict(
            tickfont=dict(size=9)
        ),
        hovermode='closest'
    )
    
    return fig

# --- Streamlit UI ---
st.title("📅 Historical Events Timeline Explorer")
st.markdown("Explore historical events from DBPedia including wars, births, and organization foundings.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Timeline Settings")
    
    # Year range selector
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1800, max_value=2023, value=1900, step=1)
    with col2:
        end_year = st.number_input("End Year", min_value=1801, max_value=2024, value=2000, step=1)
    
    if start_year >= end_year:
        st.error("End year must be greater than start year!")
    
    # Event type selector
    st.subheader("Event Types")
    event_types = st.multiselect(
        "Select event types to display:",
        ["Wars", "Births", "Organizations"],
        default=["Wars", "Births", "Organizations"]
    )
    
    # Limit selector
    limit = st.slider("Maximum number of events", min_value=50, max_value=500, value=200, step=50)
    
    # Query button
    query_button = st.button("🔍 Query DBPedia", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This app queries DBPedia's SPARQL endpoint to retrieve and visualize historical events.")

# Main content area
if query_button:
    if not event_types:
        st.warning("⚠️ Please select at least one event type.")
    elif start_year >= end_year:
        st.error("❌ Invalid date range. Please adjust the years.")
    else:
        with st.spinner(f"Querying DBPedia for events between {start_year}-{end_year}..."):
            try:
                event_data = get_timeline_data(start_year, end_year, limit, event_types)
                
                if not event_data.empty:
                    # Display statistics
                    st.success(f"✅ Successfully loaded {len(event_data)} events!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Events", len(event_data))
                    with col2:
                        st.metric("Wars", len(event_data[event_data['Type'] == 'War']))
                    with col3:
                        st.metric("Births", len(event_data[event_data['Type'] == 'Birth']))
                    with col4:
                        st.metric("Foundings", len(event_data[event_data['Type'] == 'Founding']))
                    
                    # Display timeline
                    st.plotly_chart(create_timeline(event_data), use_container_width=True)
                    
                    # Display data table
                    with st.expander("📊 View Raw Data"):
                        st.dataframe(
                            event_data[['Task', 'Start', 'Finish', 'Type']].sort_values('Start'),
                            use_container_width=True
                        )
                    
                    # Download option
                    csv = event_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"timeline_data_{start_year}_{end_year}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No data returned from DBPedia. Try adjusting your filters or check back later.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred while querying DBPedia:")
                st.exception(e)
                st.info("💡 Tip: DBPedia might be temporarily unavailable. Please try again later.")
else:
    # Welcome screen
    st.info("👈 Configure your timeline settings in the sidebar and click 'Query DBPedia' to begin.")
    
    st.markdown("### Features:")
    st.markdown("""
    - 🎯 **Customizable Date Range**: Select any period from 1800 to 2024
    - 🎨 **Filter Event Types**: Choose between Wars, Births, and Organization Foundings
    - 📊 **Interactive Visualization**: Hover over events to see details
    - 💾 **Export Data**: Download results as CSV for further analysis
    - ⚡ **Cached Queries**: Fast retrieval of previously queried data
    """)
    
    st.markdown("### Example Queries:")
    st.markdown("""
    - **World Wars Era**: 1914-1945, Wars only
    - **20th Century**: 1900-2000, All event types
    - **Post-War Period**: 1945-2000, Organizations and Births
    """)