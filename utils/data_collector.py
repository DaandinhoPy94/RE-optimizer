
import pandas as pd
import requests

class VastgoedDataCollector:
    """
    Verzamelt vastgoeddata van publieke bronnen
    """
    
    def __init__(self):
        self.cbs_base_url = "https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS"
    
    def get_gemiddelde_woningprijzen(self):
        """
        Haalt gemiddelde woningprijzen per gemeente
        CBS Dataset: 83625NED
        """
        # CBS API URL voor woningprijzen
        url = "https://opendata.cbs.nl/ODataApi/odata/83625NED/TableDataProperties"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Converteer naar DataFrame
            df = pd.DataFrame(data['value'])
            
            # Data cleaning
            df = df[df['GemiddeldeWoningwaarde'] > 0]
            
            return df
            
        except Exception as e:
            print(f"Error bij ophalen data: {e}")
            return None
    
    def get_demografische_data(self):
        """
        Haalt demografische data op per gemeente
        Nuttig voor prijsvoorspellingen
        """
        # Implementatie voor demografische data
        pass

    def get_historical_price_data(self, years_back=5):
        """
        Haalt historische woningprijzen op voor portfolio optimalisatie
        CBS Dataset: 83625NED - tijdreeksen
        """
        url = "https://opendata.cbs.nl/ODataApi/odata/83625NED/TypedDataSet"
        
        # Parameters voor filtering
        start_year = 2024 - years_back
        params = {
            '$filter': f"Perioden ge '{start_year}JJ00'",
            '$select': 'Perioden,RegioS,GemiddeldeVerkoopprijs_1'  # Correcte namen!
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse de data met correcte kolomnamen
            records = []
            for item in data.get('value', []):
                records.append({
                    'jaar': item.get('Perioden'),
                    'regio': item.get('RegioS'),
                    'gemiddelde_prijs': item.get('GemiddeldeVerkoopprijs_1')
                })
            
            df = pd.DataFrame(records)
            
            # Clean de data
            df = df[df['gemiddelde_prijs'].notna()]
            df['gemiddelde_prijs'] = pd.to_numeric(df['gemiddelde_prijs'], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error bij CBS historical data: {e}")
            return None