import pandas as pd
import plotly.express as px
from SPARQLWrapper import SPARQLWrapper, JSON
import webbrowser
import os

def get_timeline_data():
    """Queries DBPedia for historical events and returns a DataFrame."""
    
    endpoint_url = "https://dbpedia.org/sparql"
    
    # This query finds Wars, Births, and Organization Foundings
    # between the years 1900 and 2000.
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?label ?startDate ?endDate ?type
    WHERE {
      # Set the date range for the 100-year period
      FILTER(?startDate >= "1900-01-01"^^xsd:date && ?startDate < "2000-01-01"^^xsd:date)
      
      {
        # Block 1: Get Military Conflicts (Wars)
        SELECT ?label ?startDate ?endDate ("War" as ?type)
        WHERE {
          ?event a dbo:MilitaryConflict ;
                 rdfs:label ?label ;
                 dbo:startDate ?startDate ;
                 dbo:endDate ?endDate .
          FILTER (lang(?label) = "en")
        }
      }
      UNION
      {
        # Block 2: Get Births of famous people
        SELECT ?label ?startDate ?endDate ("Birth" as ?type)
        WHERE {
          ?event a dbo:Person ;
                 rdfs:label ?label ;
                 dbo:birthDate ?startDate .
          # For single-day events, set endDate = startDate
          BIND(?startDate as ?endDate) 
          FILTER (lang(?label) = "en")
        }
      }
      UNION
      {
        # Block 3: Get founding dates of organizations
        SELECT ?label ?startDate ?endDate ("Founding" as ?type)
        WHERE {
          ?event a dbo:Organisation ;
                 rdfs:label ?label ;
                 dbo:formationDate ?startDate .
          # For single-day events, set endDate = startDate
          BIND(?startDate as ?endDate)
          FILTER (lang(?label) = "en")
        }
      }
    }
    ORDER BY ?startDate
    LIMIT 200
    """

    print("Querying DBPedia... This may take a moment.")
    
    # Set up and execute the SPARQL query
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # --- Process results into a pandas DataFrame ---
    processed_results = []
    for result in results["results"]["bindings"]:
        processed_results.append({
            'Task': result['label']['value'],
            'Start': result['startDate']['value'],
            'Finish': result['endDate']['value'],
            'Type': result['type']['value']
        })
    
    if not processed_results:
        return pd.DataFrame() # Return empty frame if no results

    df = pd.DataFrame(processed_results)

    # **Crucial Step:** Convert date strings to datetime objects
    # Errors='coerce' will turn any bad dates into NaT (Not a Time)
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Finish'] = pd.to_datetime(df['Finish'], errors='coerce')
    
    # Drop any rows where the Start date couldn't be parsed
    df = df.dropna(subset=['Start'])
    
    # If Finish is bad (NaT), set it to be the same as Start
    df['Finish'] = df['Finish'].fillna(df['Start'])
    
    # ------------------------------------------------------------------
    # --- THIS IS THE FIX ---
    # Find all events where Start and Finish are the same
    mask = (df['Finish'] == df['Start'])
    
    # For those events, add 1 day to the Finish time to make them visible
    df.loc[mask, 'Finish'] = df.loc[mask, 'Start'] + pd.DateOffset(days=1)
    # ------------------------------------------------------------------
    
    return df
    
def create_timeline(df):
    """Creates and displays an interactive timeline chart from the DataFrame."""
    
    print(f"Loaded {len(df)} events. Creating timeline...")

    # Use Plotly Express's timeline function (which is a Gantt chart)
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Task",           # Show the event label on the y-axis
        color="Type",       # Color-code the bars by our ?type variable
        hover_name="Task",  # Show task name clearly on hover
        title="Historical Events Timeline (1900-2000)"
    )

    # Improve layout:
    # - Reverse y-axis to show earlier events at the top
    # - Set a fixed height
    # - Make y-axis labels smaller to fit more events
    fig.update_yaxes(autorange="reversed") 
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Event",
        height=900,
        title_x=0.5,
        yaxis=dict(
            title_text="", # Remove y-axis title
            tickfont=dict(size=9) # Make text smaller
        )
    )
    
    # Save to an HTML file
    output_file = "historical_timeline.html"
    fig.write_html(output_file)
    
    print(f"\nSuccess! Opening timeline in your browser.")
    print(f"File saved to: {os.path.abspath(output_file)}")
    
    # Open the generated HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(output_file))

# --- Main execution block ---
if __name__ == "__main__":
    try:
        event_data = get_timeline_data()

        print(event_data)
        
        if not event_data.empty:
            create_timeline(event_data)
        else:
            print("No data returned from DBPedia.")
            print("This could be a temporary DBPedia issue or no events found for the query.")
            
    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(f"Error: {e}")
        print("Please check your internet connection and ensure all libraries are installed.")