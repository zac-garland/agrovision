library(WikidataR)

get_common_name  <- function(name){
  glue::glue('SELECT ?commonName ?lang WHERE {{
    ?item wdt:P225 "{name}".
    ?item wdt:P1843 ?commonName.
    BIND(LANG(?commonName) as ?lang)
  }}
  ORDER BY ?lang') %>% 
  query_wikidata() %>% 
  filter(lang == "en") %>% 
  slice(1) %>% 
  pull(commonName)
}

get_common_name("Lactuca") 


jsonlite::read_json("models/species_to_common_name.json") %>% 
  enframe() %>% 
  unnest(value)
