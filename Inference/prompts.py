CONSTRUCT_AUXILIARY_INFORMATION_PROMPT = '''
[Task description]: 
You should answer the {Question} in the following steps: 
<step 1> Find out what {Knowledge Planing} do you need to solve the {Question}. 
<step 2> Using the atomic relationships in {Possible Atomic Relationships}, strictly fill the {Knowledge Planing} to construct the {Knowledge Graph} as complete as possible with (Cypher) with your own knowledge.
<step 3> Strictly complete the {Knowledge Graph} to construct {Completed Knowledge Graph} to include more detailed reasoning paths that solve the {Question}.
<step 4> Based on the {Completed Knowledge Graph}, generate valid {Relation Paths} that can be helpful for answering the {Question}.


[Example 1]: 
{Question}: what is the name of justin bieber brother?
{Possible Atomic Relationships}: ['people_person_sibling_s, people_sibling_relationship_sibling', 'royalty_royal_line_preceded_by', 'people_family_name_people_with_this_family_name']

<step 1> {Knowledge Planning}: 
To answer the question "What is the name of Justin Bieber's brother?" we need to gather information about Justin Bieber's family members, specifically his siblings. Here are the key points we should include in the knowledge graph:
- Introduction:
  - Provide a brief overview of Justin Bieber and his family.
- Siblings:
  - List the names of Justin Bieber's siblings.
      
<step 2> {Knowledge Graph}: 
// Create Justin Bieber node
CREATE (justin:Person {name: "Justin Bieber"})
    
// Create Justin Bieber's brother node
CREATE (justinBrother:Person {name: "Jaxon Bieber"})

//Create temperary node for multi-hop atomic relationship 'people_person_sibling_s, people_sibling_relationship_sibling'
CREATE (tempNode:Node {name: "temp"})
    
// Connect Justin Bieber and his brother with 'people_person_sibling_s, people_sibling_relationship_sibling' relationship
CREATE (justin)-[:people_person_sibling_s]->(tempNode)
CREATE (tempNode:)-[:people_sibling_relationship_sibling]->(justinBrother)
    
<step 3> {Completed Knowledge Graph}: 
// Create Justin Bieber node
CREATE (justin:Person {name: "Justin Bieber"})
    
// Create Justin Bieber's family members node
CREATE (justinBrother:Person {name: "Jaxon Bieber"})
CREATE (justinFather:Person {name: "Jeremy Bieber"})
CREATE (justinMother:Person {name: "Pattie Mallette"})

//Create temperary node for multi-hop atomic relationship 'people_person_sibling_s, people_sibling_relationship_sibling'
CREATE (tempNode:Node {name: "temp"})
    
// Connect Justin Bieber and his brother with 'people_person_sibling_s, people_sibling_relationship_sibling' relationship
CREATE (justin)-[:people_person_sibling_s]->(tempNode)
CREATE (tempNode:)-[:people_sibling_relationship_sibling]->(justinBrother)
    
// Additional relationship help in finding Justin Bieber's brother
CREATE (justin)-[:parent]->(justinFather)
CREATE (justinFather)-[:children]->(justin)
CREATE (justinBrother)-[:parent]->(justinFather)
CREATE (justinFather)-[:children]->(justinBrother)
CREATE (justin)-[:PARENT]->(justinMother)
CREATE (justinMother)-[:children]->(justin)
CREATE (justinBrother)-[:parent]->(justinMother)
CREATE (justinMother)-[:children]->(justinBrother)

<step 4> {Relation Paths}: 
path1: [(justin)-[:people_person_sibling_s]->(tempNode)-[:people_sibling_relationship_sibling]->(justinBrother)]
path2: [(justin)-[:parent]->(justinFather)-[:children]->(justinBrother)]
path3: [(justin)-[:parent]->(justinMother)-[:children]->(justinBrother)]


[Example 2]: 
{Question}: what character did natalie portman play in star wars?
{Possible Atomic Relationships}: ['film_actor_film, film_performance_character', 'base_militaryinfiction_military_conflict_in_fiction_military_characters_involved', 'base_battlestargalactica_humanoid_cylon_model_portrayed_by']

<step 1> {Knowledge Planning}: 
To answer this question, we need information about Natalie Portman's role in the Star Wars films. The key relationships we need are:
- The films in which Natalie Portman acted.
- The characters she portrayed in those films.

<step 2> {Knowledge Graph}: 
// Create Natalie Portman node and Star Wars films
CREATE (natalie:Person {name: 'Natalie Portman'})
    
// Create character node for the role played by Natalie Portman
CREATE (padme:Character {name: 'Padmé Amidala'})

//Create temperary node for multi-hop atomic relationship 'film_actor_film, film_performance_character'
CREATE (tempNode:Node {name: "temp"})

// Connect Natalie Portman to Padmé Amidala character with 'film_actor_film, film_performance_character' relationship
CREATE (natalie)-[:film_actor_film]->(tempNode)
CREATE (tempNode)-[:film_performance_character]->(padme)

<step 3> {Completed Knowledge Graph}: 
// Create Natalie Portman node and Star Wars films
CREATE (natalie:Person {name: 'Natalie Portman'})
    
// Create character node for the role played by Natalie Portman
CREATE (padme:Character {name: 'Padmé Amidala'})

//Create temperary node for multi-hop atomic relationship 'film_actor_film, film_performance_character'
CREATE (tempNode:Node {name: "temp"})

// Connect Natalie Portman to Padmé Amidala character with 'film_actor_film, film_performance_character' relationship
CREATE (natalie)-[:film_actor_film]->(tempNode)
CREATE (tempNode)-[:film_performance_character]->(padme)

// Add specific Star Wars films where Natalie Portman appeared
CREATE (episode1:Film {title: 'Star Wars: Episode I – The Phantom Menace'})
CREATE (episode2:Film {title: 'Star Wars: Episode II – Attack of the Clones'})
CREATE (episode3:Film {title: 'Star Wars: Episode III – Revenge of the Sith'})

// Add relationships connecting Natalie Portman to these specific films
CREATE (natalie)-[:acted_in]->(episode1)
CREATE (natalie)-[:acted_in]->(episode2)
CREATE (natalie)-[:acted_in]->(episode3)

<step 4> {Relation Paths}: 
path1: [(natalie)-[:film_actor_film]->(tempNode)-[:film_performance_character]->(padme)]
path2: [(natalie)-[:acted_in]->(episode1), (natalie)-[:acted_in]->(episode2), (natalie)-[:acted_in]->(episode3)]


[Example 3]: 
{Question}: what country is the grand bahama island in?
{Possible Atomic Relationships}: ['base_locations_countries_places_within', 'location.location.containedby', 'base_locations_place_in_the_world_country']

<step 1> {Knowledge Planning}: 
To answer this question, we need information about the geographic location of Grand Bahama Island and its relation to a country. Specifically, we need:
- The geographical entity of Grand Bahama Island.
- The country it is contained in.

<step 2> {Knowledge Graph}: 
// Create Grand Bahama Island and country node
CREATE (grandBahama:Location {name: 'Grand Bahama Island'})
CREATE (bahamas:Country {name: 'Bahamas'})

// Connect Grand Bahama Island to Bahamas with 'base_locations_countries_places_within' and 'location.location.containedby' relationships
CREATE (grandBahama)-[:base_locations_countries_places_within]->(bahamas)
CREATE (grandBahama)-[:location_location_containedby]->(bahamas)

<step 3> {Completed Knowledge Graph}: 
// Create Grand Bahama Island and country node
CREATE (grandBahama:Location {name: 'Grand Bahama Island'})
CREATE (bahamas:Country {name: 'Bahamas'})

// Connect Grand Bahama Island to Bahamas with 'base_locations_countries_places_within' and 'location.location.containedby' relationships
CREATE (grandBahama)-[:base_locations_countries_places_within]->(bahamas)
CREATE (grandBahama)-[:location_location_containedby]->(bahamas)

// Create additional places within the Bahamas
CREATE (nassau:Location {name: 'Nassau'})
CREATE (andros:Location {name: 'Andros Island'})

// Connect other places to the Bahamas
CREATE (grandBahama)-[:location_location_containedby]->(bahamas)
CREATE (bahamas))-[:contains]->(nassau)

<step 4> {Relation Paths}: 
path1: [(grandBahama)-[:base_locations_countries_places_within]->(bahamas)]
path2: [(grandBahama)-[:location_location_containedby]->(bahamas)]
path3: [(grandBahama)-[:location_location_containedby]->(bahamas)-[:contains]->(bahamas)]


[Example 4]: 
{Question}: where are the nfl redskins from?
{Possible Atomic Relationships}: ['freebase_type_kind_types', 'freebase_query_hints_related_domain', 'sports_sports_team_location']

<step 1> {Knowledge Planning}: 
To answer this question, we require the information about NFL Redskins:
- The name of the team: NFL Redskins.
- The geographical location or city associated with the team.

<step 2> {Knowledge Graph}: 
// Create node for the NFL Redskins (now Washington Commanders)
CREATE (redskins:SportsTeam {name: "Washington Redskins"})

// Create node for the location (Washington, D.C.)
CREATE (washingtonDC:Location {name: "Washington, D.C."})

// Create relationship connecting the team to their location
CREATE (redskins)-[:sports_sports_team_location]->(washingtonDC)

<step 3> {Completed Knowledge Graph}: 
// Create node for the NFL Redskins (now Washington Commanders)
CREATE (redskins:SportsTeam {name: "Washington Redskins"})

// Create node for the location (Washington, D.C.)
CREATE (washingtonDC:Location {name: "Washington, D.C."})

// Create node for the current team name (Washington Commanders)
CREATE (commanders:SportsTeam {name: "Washington Commanders"})

// Create relationship connecting the team to their location
CREATE (redskins)-[:sports_sports_team_location]->(washingtonDC)
CREATE (commanders)-[:sports_sports_team_location]->(washingtonDC)

<step 4> {Relation Paths}: 
Path1: [(redskins)-[:sports_sports_team_location]->(washingtonDC)]
Path2: [(commanders)-[:sports_sports_team_location]->(washingtonDC)]


[Example 5]: 
{Question}: who starred in the movies directed by Jan Egleson?
{Possible Atomic Relationships}: ['film.director.film', 'film_content_rating_film', 'film_person_or_entity_appearing_in_film_films, film_personal_film_appearance_film', 'film_actor_film, film_performance_film', 'tv_tv_actor_starring_roles, tv_regular_tv_appearance_series', 'film_film_starring, film_performance_actor']

<step 1> {Knowledge Planning}: 
To solve this question, we need to gather information regarding:
- The films directed by Jan Egleson.
- The actors or entities who appeared in or starred in those films.

<step 2> {Knowledge Graph}: 
// Create a node for Jan Egleson
CREATE (jan:Person {name: "Jan Egleson"})

// Create a node for a movie directed by Jan Egleson
CREATE (movie1:Film {title: "The Dark End of the Street"})
CREATE (movie2:Film {title: "A Shock to the System"})

// Create relationships connecting Jan Egleson to the films he directed
CREATE (jan)-[:film_director_film]->(movie1)
CREATE (jan)-[:film_director_film]->(movie2)

// Create nodes for actors starring in these films
CREATE (actor1:Person {name: "Ben Affleck"})
CREATE (actor2:Person {name: "Michael Caine"})

//Create temperary node for multi-hop atomic relationship 'film_film_starring, film_performance_actor'
CREATE (tempNode:Node {name: "temp"})

// Connect actors to the films they starred in
CREATE (movie1)-[:film_film_starring]->(tempNode)
CREATE (tempNode)-[:film_performance_actor]->(actor1)
CREATE (movie2)-[:film_film_starring]->(tempNode)
CREATE (tempNode)-[:film_performance_actor]->(actor2)

<step 3> {Completed Knowledge Graph}: 
// Create a node for Jan Egleson
CREATE (jan:Person {name: "Jan Egleson"})

// Create a node for a movie directed by Jan Egleson
CREATE (movie1:Film {title: "The Dark End of the Street"})
CREATE (movie2:Film {title: "A Shock to the System"})

// Create relationships connecting Jan Egleson to the films he directed
CREATE (jan)-[:film_director_film]->(movie1)
CREATE (jan)-[:film_director_film]->(movie2)

// Create nodes for actors starring in these films
CREATE (actor1:Person {name: "Ben Affleck"})
CREATE (actor2:Person {name: "Michael Caine"})

//Create temperary node for multi-hop atomic relationship 'film_film_starring, film_performance_actor'
CREATE (tempNode:Node {name: "temp"})

// Connect actors to the films they starred in
CREATE (movie1)-[:film_film_starring]->(tempNode)
CREATE (tempNode)-[:film_performance_actor]->(actor1)
CREATE (movie2)-[:film_film_starring]->(tempNode)
CREATE (tempNode)-[:film_performance_actor]->(actor2)

// Create additional actor starring in these films
CREATE (actor3:Person {name: "William H. Macy"})

// Connect additional actors to the films they starred in
CREATE (movie1)-[:film_film_starring]->(tempNode)
CREATE (tempNode)-[:film_performance_actor]->(actor3)

<step 4> {Relation Paths}: 
path1: [(jan)-[:film_director_film]->(movie1)-[:film_film_starring]->(tempNode)-[:film_performance_actor]->(actor1)]
path2: [(jan)-[:film_director_film]->(movie2)-[:film_film_starring]->(tempNode)-[:film_performance_actor]->(actor2)]
path3: [(jan)-[:film_director_film]->(movie1)-[:film_film_starring]->(tempNode)-[:film_performance_actor]->(actor3)]


[Example 6]:
{Question}: who wrote the novel 1984?
{Possible Atomic Relationships}: ['literary_author_written_work', 'people_written_works', 'literary_written_works_author']

<step 1> {Knowledge Planning}:  
To answer this question, we need to gather the following details:
- Information about the author of *1984*.
- The work that the author is associated with.

<step 2> {Knowledge Graph}:  
// Create node for the novel 1984
CREATE (novel:Book {title: "1984"})  
  
// Create node for the author George Orwell  
CREATE (orwell:Person {name: "George Orwell"})  
  
// Create relationship connecting George Orwell to 1984
CREATE (orwell)-[:literary_author_written_work]->(novel)

<step 3> {Completed Knowledge Graph}:  
// Create node for the novel 1984 
CREATE (novel:Book {title: "1984"})  
  
// Create node for the author George Orwell  
CREATE (orwell:Person {name: "George Orwell"})  
  
// Create relationship connecting George Orwell to 1984  
CREATE (orwell)-[:literary_author_written_work]->(novel)  

<step 4> {Relation Paths}:  
path1: [(orwell)-[:literary_author_written_work]->(novel)]


[Task]: 
{Question}: %s
{Possible Atomic Relationships}: %s'''


CONSTRUCT_AUXILIARY_INFORMATION_PROMPT_CWQ = '''
[Task description]: 
You should answer the {Question} in the following steps: 
<step 1> Find out what {Knowledge Planing} do you need to solve the {Question}. 
<step 2> Using the atomic relationships in {Possible Atomic Relationships}, strictly fill the {Knowledge Planing} to construct the {Knowledge Graph} as complete as possible with (Cypher) with your own knowledge.
<step 3> Strictly complete the {Knowledge Graph} to construct {Completed Knowledge Graph} to include more detailed reasoning paths that solve the {Question}.
<step 4> Based on the {Completed Knowledge Graph}, generate valid {Relation Paths} that can be helpful for answering the {Question}.


[Example 1]: 
{Question}: what is the name of justin bieber brother?
{Possible Atomic Relationships}: ['people_person_sibling_s, people_sibling_relationship_sibling', 'royalty_royal_line_preceded_by', 'people_family_name_people_with_this_family_name']

<step 1> {Knowledge Planning}: 
To answer the question "What is the name of Justin Bieber's brother?" we need to gather information about Justin Bieber's family members, specifically his siblings. Here are the key points we should include in the knowledge graph:
- Introduction:
  - Provide a brief overview of Justin Bieber and his family.
- Siblings:
  - List the names of Justin Bieber's siblings.
      
<step 2> {Knowledge Graph}: 
// Create Justin Bieber node
CREATE (justin:Person {name: "Justin Bieber"})
    
// Create Justin Bieber's brother node
CREATE (justinBrother:Person {name: "Jaxon Bieber"})

//Create temperary node for multi-hop atomic relationship 'people_person_sibling_s, people_sibling_relationship_sibling'
CREATE (tempNode:Node {name: "temp"})
    
// Connect Justin Bieber and his brother with 'people_person_sibling_s, people_sibling_relationship_sibling' relationship
CREATE (justin)-[:people_person_sibling_s]->(tempNode)
CREATE (tempNode:)-[:people_sibling_relationship_sibling]->(justinBrother)
    
<step 3> {Completed Knowledge Graph}: 
// Create Justin Bieber node
CREATE (justin:Person {name: "Justin Bieber"})
    
// Create Justin Bieber's family members node
CREATE (justinBrother:Person {name: "Jaxon Bieber"})
CREATE (justinFather:Person {name: "Jeremy Bieber"})
CREATE (justinMother:Person {name: "Pattie Mallette"})

//Create temperary node for multi-hop atomic relationship 'people_person_sibling_s, people_sibling_relationship_sibling'
CREATE (tempNode:Node {name: "temp"})
    
// Connect Justin Bieber and his brother with 'people_person_sibling_s, people_sibling_relationship_sibling' relationship
CREATE (justin)-[:people_person_sibling_s]->(tempNode)
CREATE (tempNode:)-[:people_sibling_relationship_sibling]->(justinBrother)
    
// Additional relationship help in finding Justin Bieber's brother
CREATE (justin)-[:parent]->(justinFather)
CREATE (justinFather)-[:children]->(justin)
CREATE (justinBrother)-[:parent]->(justinFather)
CREATE (justinFather)-[:children]->(justinBrother)
CREATE (justin)-[:PARENT]->(justinMother)
CREATE (justinMother)-[:children]->(justin)
CREATE (justinBrother)-[:parent]->(justinMother)
CREATE (justinMother)-[:children]->(justinBrother)

<step 4> {Relation Paths}: 
path1: [(justin)-[:people_person_sibling_s]->(tempNode)-[:people_sibling_relationship_sibling]->(justinBrother)]
path2: [(justin)-[:parent]->(justinFather)-[:children]->(justinBrother)]
path3: [(justin)-[:parent]->(justinMother)-[:children]->(justinBrother)]


[Example 2]: 
{Question}: What movie with film character named Mr. Woodson did Tupac star in?  
{Possible Atomic Relationships}: ['film_film_starring, film_performance_character', 'film_actor_film', 'film_film_character_portrayed_in_films, film_performance_actor', 'film_film_character_portrayed_in_films, film_performance_film', 'film_actor_film, film_performance_character', 'film_actor_film, film_performance_film']

<step 1> {Knowledge Planning}: 
To answer the question, we need to gather information about:
- Tupac Shakur’s filmography.
- Identify the character named Mr. Woodson.
- Determine which movie features this character and Tupac.

<step 2> {Knowledge Graph}: 
// Create Tupac node
CREATE (tupac:Person {name: "Tupac Shakur"})

// Create Mr. Woodson node
CREATE (mrWoodson:Character {name: "Mr. Woodson"})

// Create the movie node
CREATE (movie:Film {title: "Gridlock'd"})

//Create temperary node for multi-hop atomic relationship 'film_actor_film, film_performance_film'
CREATE (tempNode1:Node {name: "temp"})

//Create temperary node for multi-hop atomic relationship 'film_film_starring, film_performance_character'
CREATE (tempNode2:Node {name: "temp"})

// Connect Tupac to the movie with 'film_actor_film, film_performance_film' relationship
CREATE (tupac)-[:film_actor_film]->(tempNode1)
CREATE (tempNode1)-[:film_performance_film]->(movie)

// Connect the movie to the character Mr. Woodson with 'film_film_starring, film_performance_character' relationship
CREATE (movie)-[:film_film_starring]->(tempNode2)
CREATE (tempNode2)-[:film_performance_character]->(mrWoodson)

<step 3> {Completed Knowledge Graph}: 
// Create Tupac node
CREATE (tupac:Person {name: "Tupac Shakur"})

// Create Mr. Woodson node
CREATE (mrWoodson:Character {name: "Mr. Woodson"})

// Create the movie node
CREATE (movie:Film {title: "Gridlock'd"})

// Create additional characters and details about the film
CREATE (otherCharacter:Character {name: "Robbin'")})

//Create temperary node for multi-hop atomic relationship 'film_actor_film, film_performance_film'
CREATE (tempNode1:Node {name: "temp"})

//Create temperary node for multi-hop atomic relationship 'film_film_starring, film_performance_character'
CREATE (tempNode2:Node {name: "temp"})

// Connect Tupac to the movie with 'film_actor_film, film_performance_film' relationship
CREATE (tupac)-[:film_actor_film]->(tempNode1)
CREATE (tempNode1)-[:film_performance_film]->(movie)

// Connect the movie to the character Mr. Woodson with 'film_film_starring, film_performance_character' relationship
CREATE (movie)-[:film_film_starring]->(tempNode2)
CREATE (tempNode2)-[:film_performance_character]->(mrWoodson)

// Connect the movie to the character Robbin with 'film_film_starring, film_performance_character' relationship
CREATE (movie)-[:film_film_starring]->(tempNode2)
CREATE (tempNode2)-[:film_performance_character]->(otherCharacter)

<step 4> {Relation Paths}: 
path1: [(tupac)-[:film_actor_film]->(tempNode1)--[:film_performance_film]->(movie)-[:film_film_starring]->(tempNode2)-[:film_performance_character]->(mrWoodson)]
path2: [(tupac)-[:film_actor_film]->(tempNode1)--[:film_performance_film]->(movie)-[:film_film_starring]->(tempNode2)-[:film_performance_character]->(otherCharacter)]


[Example 3]: 
{Question}: What country sharing borders with Spain does the Setúbal District belong to?  
{Possible Atomic Relationships}: ['base_locations_countries_states_provinces_within', 'base_locations_countries_places_within', 'location_location_adjoin_s, location_adjoining_relationship_adjoins', 'base_locations_countries_counties_within', 'location_country_administrative_divisions', 'base_locations_continents_states_provinces_within']

<step 1> {Knowledge Planning}: 
To answer this question, we need to gather information about:
- The Setúbal District's location.
- The country that contains the Setúbal District.
- Countries that share borders with Spain.

<step 2> {Knowledge Graph}: 
// Create Spain node
CREATE (spain:Country {name: "Spain"})

// Create Portugal node
CREATE (portugal:Country {name: "Portugal"})

// Create Setúbal District node
CREATE (setubal:Place {name: "Setúbal District"})

//Create temperary node for multi-hop atomic relationship 'location_location_adjoin_s, location_adjoining_relationship_adjoins'
CREATE (tempNode:Node {name: "temp"})

// Connect Spain to Portugal with 'location_location_adjoin_s, location_adjoining_relationship_adjoins' relationship
CREATE (spain)-[:location_location_adjoin_s]->(tempNode)
CREATE (tempNode)-[:location_adjoining_relationship_adjoins]->(portugal)

// Connect Portugal to Setúbal District with 'location_country_administrative_divisions' relationship
CREATE (setubal)-[:location_country_administrative_divisions]->(portugal)

<step 3> {Completed Knowledge Graph}: 
// Create Spain node
CREATE (spain:Country {name: "Spain"})

// Create Portugal node
CREATE (portugal:Country {name: "Portugal"})

// Create Setúbal District node
CREATE (setubal:Place {name: "Setúbal District"})

//Create temperary node for multi-hop atomic relationship 'location_location_adjoin_s, location_adjoining_relationship_adjoins'
CREATE (tempNode:Node {name: "temp"})

// Connect Spain to Portugal with 'location_location_adjoin_s, location_adjoining_relationship_adjoins' relationship
CREATE (spain)-[:location_location_adjoin_s]->(tempNode)
CREATE (tempNode)-[:location_adjoining_relationship_adjoins]->(portugal)

// Connect Portugal to Setúbal District with 'location_country_administrative_divisions' relationship
CREATE (portugal)-[:location_country_administrative_divisions]->(setubal)

// Add more countries sharing borders with Spain for context
CREATE (france:Country {name: "France"})
CREATE (gibraltar:Place {name: "Gibraltar"})

// Connect Spain to its neighboring countries
CREATE (spain)-[:location_location_adjoin_s]->(tempNode)
CREATE (tempNode)-[:location_adjoining_relationship_adjoins]->(france)
CREATE (spain)-[:location_location_adjoin_s]->(tempNode)
CREATE (tempNode)-[:location_adjoining_relationship_adjoins]->(gibraltar)

<step 4> {Relation Paths}: 
path1: [(spain)-[:location_location_adjoin_s]->(tempNode)-[:location_adjoining_relationship_adjoins]->(portugal)-[:location_country_administrative_divisions]->(setubal)]
path2: [(spain)-[:location_location_adjoin_s]->(tempNode)-[:location_adjoining_relationship_adjoins]->(france)-[:location_country_administrative_divisions]->(setubal)]
path3: [(spain)-[:location_location_adjoin_s]->(tempNode)-[:location_adjoining_relationship_adjoins]->(gibraltar)-[:location_country_administrative_divisions]->(setubal)]


[Example 4]: 
{Question}: What is the name of the president of the geographic location where Nicolas Sarkozy was appointed to a governmental position?  
{Possible Atomic Relationships}: ['government_governmental_body_members, government_government_position_held_office_holder', 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by', 'government_governmental_jurisdiction_governing_officials, government_governmental_jurisdiction_government_positions', 'government_governmental_jurisdiction_governing_officials, government_government_position_held_office_holder', 'base_locations_place_in_the_world_county', 'location_hud_county_place_place']

<step 1> {Knowledge Planning}: 
To answer the question, we need to gather information about:
- Nicolas Sarkozy's governmental positions.
- The geographical location related to those positions.
- The current president of that location.

<step 2> {Knowledge Graph}: 
// Create Nicolas Sarkozy node
CREATE (sarkozy:Person {name: "Nicolas Sarkozy"})

// Create node for France
CREATE (france:Country {name: "France"})

// Create node for the current president
CREATE (currentPresident:Person {name: "Emmanuel Macron"})

//Create temperary node for multi-hop atomic relationship 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by'
CREATE (tempNode1:Node {name: "temp"})

//Create temperary node for multi-hop atomic relationship 'government_governmental_jurisdiction_governing_officials, government_government_position_held_office_holder'
CREATE (tempNode2:Node {name: "temp"})

// Connect Frace to Nicolas Sarkozy with 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by' relationship
CREATE (france)-[:government_governmental_jurisdiction_governing_officials]->(tempNode1)
CREATE (tempNode1)-[:government_government_position_held_appointed_by]->(sarkozy)

// Connect France to current president with 'government_governmental_jurisdiction_governing_officials, government_government_position_held_office_holder' relationship
CREATE (france)-[:government_governmental_jurisdiction_governing_officials]->(tempNode2)
CREATE (tempNode2)-[:government_government_position_held_office_holder]->(currentPresident)

<step 3> {Completed Knowledge Graph}: 
// Create Nicolas Sarkozy node
CREATE (sarkozy:Person {name: "Nicolas Sarkozy"})

// Create node for France
CREATE (france:Country {name: "France"})

// Create node for the current president
CREATE (currentPresident:Person {name: "Emmanuel Macron"})

//Create temperary node for multi-hop atomic relationship 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by'
CREATE (tempNode1:Node {name: "temp"})

//Create temperary node for multi-hop atomic relationship 'government_governmental_jurisdiction_governing_officials, government_government_position_held_office_holder'
CREATE (tempNode2:Node {name: "temp"})

// Connect Frace to Nicolas Sarkozy with 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by' relationship
CREATE (france)-[:government_governmental_jurisdiction_governing_officials]->(tempNode1)
CREATE (tempNode1)-[:government_government_position_held_appointed_by]->(sarkozy)

// Connect France to current president with 'government_governmental_jurisdiction_governing_officials, government_government_position_held_office_holder' relationship
CREATE (france)-[:government_governmental_jurisdiction_governing_officials]->(tempNode2)
CREATE (tempNode2)-[:government_government_position_held_office_holder]->(currentPresident)

// Create node for the position Nicolas Sarkozy held
CREATE (position:GovernmentPosition {title: "President of France"})

// Connect Nicolas Sarkozy to his governmental position
CREATE (sarkozy)-[:government_government_position_held_office_position_or_title]->(position)

<step 4> {Relation Paths}: 
path1: [(france)-[:government_governmental_jurisdiction_governing_officials]->(tempNode1)-[:government_government_position_held_appointed_by]->(sarkozy), (france)-[:government_governmental_jurisdiction_governing_officials]->(tempNode2)-[:government_government_position_held_office_holder]->(currentPresident)]
path2: [(sarkozy)-[:government_government_position_held_office_position_or_title]->(position)]


[Example 5]: 
{Question}: What educational institution with mascot named Yale University Handsome Dan did James Franco go to?
{Possible Atomic Relationships}: ['people_person_education, education_education_institution', 'education_educational_institution_parent_institution', 'education_educational_institution_campus_educational_institution', 'education_education_institution', 'education_educational_institution_mascot', 'education_educational_institution_school_type']

<step 1> {Knowledge Planning}: 
To answer the question about where James Franco studied that has a mascot named Yale University Handsome Dan, we need to:
- Identify James Franco's educational history.
- Determine which institution he attended that matches the mascot description.
- Confirm that this institution is indeed Yale University or related to it.

<step 2> {Knowledge Graph}: 
// Create James Franco node
CREATE (james:Person {name: "James Franco"})

// Create Yale University node
CREATE (yale:University {name: "Yale University"})

// Create Handsome Dan node
CREATE (handsomeDan:Mascot {name: "Handsome Dan"})

//Create temporary node for multi-hop atomic relationship 'people_person_education, education_education_institution'
CREATE (tempNode:Node {name: "temp"})

// Connect James Franco to Yale with 'people_person_education, education_education_institution' relationship
CREATE (james)-[:people_person_education]->(tempNode)
CREATE (tempNode)-[:education_education_institution]->(yale)

// Connect Yale to its mascot
CREATE (yale)-[:education_educational_institution_mascot]->(handsomeDan)

<step 3> {Completed Knowledge Graph}: 
// Create James Franco node
CREATE (james:Person {name: "James Franco"})

// Create Yale University node
CREATE (yale:University {name: "Yale University"})

// Create Handsome Dan node
CREATE (handsomeDan:Mascot {name: "Handsome Dan"})

//Create temporary node for multi-hop atomic relationship 'people_person_education, education_education_institution'
CREATE (tempNode:Node {name: "temp"})

// Connect James Franco to Yale with 'people_person_education, education_education_institution' relationship
CREATE (james)-[:people_person_education]->(tempNode)
CREATE (tempNode)-[:education_education_institution]->(yale)

// Connect Yale to its mascot
CREATE (yale)-[:education_educational_institution_mascot]->(handsomeDan)

// Additional nodes for other universities James Franco attended for context
CREATE (ucla:University {name: "UCLA"})
CREATE (nyu:University {name: "NYU"})

// Connect James Franco to other universities he attended
CREATE (james)-[:people_person_education]->(tempNode)
CREATE (tempNode)-[:education_education_institution]->(ucla)
CREATE (james)-[:people_person_education]->(tempNode)
CREATE (tempNode)-[:education_education_institution]->(nyu)

<step 4> {Relation Paths}: 
path1: [(james)-[:people_person_education]->(tempNode)-[:education_education_institution]->(yale)-[:education_educational_institution_mascot]->(handsomeDan)]
path2: [(james)-[:people_person_education]->(tempNode)-[:education_education_institution]->(ucla)-[:education_educational_institution_mascot]->(handsomeDan)]
path3: [(james)-[:people_person_education]->(tempNode)-[:education_education_institution]->(nyu)-[:education_educational_institution_mascot]->(handsomeDan)]


[Example 6]: 
{Question}: What language is spoken in the location that appointed Margrethe II of Denmark to a governmental position?
{Possible Atomic Relationships}: ['government_governmental_body_offices_positions', 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by', 'government_governmental_body_members, government_government_position_held_office_holder', 'location_country_languages_spoken', 'language_human_language_countries_spoken_in', 'people_ethnicity_languages_spoken']

<step 1> {Knowledge Planning}: 
To answer this question, we need to gather information about:
- Margrethe II of Denmark's governmental position
- The location/country that appointed her
- The official language(s) spoken in that location
- The relationship between the location and its languages

<step 2> {Knowledge Graph}: 
// Create Margrethe II node
CREATE (margrethe:Person {name: "Margrethe II of Denmark"})

// Create Denmark node
CREATE (denmark:Country {name: "Denmark"})

// Create Danish language node
CREATE (danish:Language {name: "Danish language"})

//Create temperary node for multi-hop atomic relationship 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by'
CREATE (tempNode:Node {name: "temp"})

// Connect Denmark to Margrethe II with 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by' relationship
CREATE (denmark)-[:government_governmental_jurisdiction_governing_officials]->(tempNode)
CREATE (tempNode)-[:government_government_position_held_appointed_by]->(margrethe)

// Connect Denmark to Danish language with 'location_country_languages_spoken' relationship
CREATE (denmark)-[:location_country_languages_spoken]->(danish)

<step 3> {Completed Knowledge Graph}: 
// Create Margrethe II node
CREATE (margrethe:Person {name: "Margrethe II of Denmark"})

// Create Denmark node
CREATE (denmark:Country {name: "Denmark"})

// Create language nodes
CREATE (danish:Language {name: "Danish language"})
CREATE (greenlandic:Language {name: "Greenlandic language"})
CREATE (german:Language {name: "German language"})

//Create temperary node for multi-hop atomic relationship 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by'
CREATE (tempNode:Node {name: "temp"})

// Connect Denmark to Margrethe II with 'government_governmental_jurisdiction_governing_officials, government_government_position_held_appointed_by' relationship
CREATE (denmark)-[:government_governmental_jurisdiction_governing_officials]->(tempNode)
CREATE (tempNode)-[:government_government_position_held_appointed_by]->(margrethe)

// Connect Denmark to languages with 'location_country_languages_spoken' relationship
CREATE (denmark)-[:location_country_languages_spoken]->(danish)
CREATE (denmark)-[:location_country_languages_spoken]->(greenlandic)
CREATE (denmark)-[:location_country_languages_spoken]->(german)

<step 4> {Relation Paths}: 
path1: [(denmark)-[:government_governmental_jurisdiction_governing_officials]->(tempNode)-[:government_government_position_held_appointed_by]->(margrethe), (denmark)-[:location_country_languages_spoken]->(danish)]
path2: [(denmark)-[:government_governmental_jurisdiction_governing_officials]->(tempNode)-[:government_government_position_held_appointed_by]->(margrethe), (denmark)-[:location_country_languages_spoken]->(greenlandic)]
path3: [(denmark)-[:government_governmental_jurisdiction_governing_officials]->(tempNode)-[:government_government_position_held_appointed_by]->(margrethe), (denmark)-[:location_country_languages_spoken]->(german)]


[Example 7]: 
{Question}: What country did Prith Banerjee in that uses United States Dollar?
{Possible Atomic Relationships}: ['people_person_places_lived, people_place_lived_location', 'people_person_places_lived', 'location_location_people_born_here', 'location_imports_and_exports_imported_from', 'medicine_drug_legal_status_country', 'location_country_currency_used']

<step 1> {Knowledge Planning}: 
To answer the question, we need to gather information about:
- Prith Banerjee's places of residence or any specific country he has been associated with.
- The currency used in that country, specifically looking for the United States Dollar.
- Confirm that this country is indeed one that uses the USD.

<step 2> {Knowledge Graph}: 
// Create Prith Banerjee node
CREATE (prith:Person {name: "Prith Banerjee"})

// Create United States node
CREATE (usa:Country {name: "United States"})

// Create a node for the currency
CREATE (usd:Currency {name: "United States Dollar"})

// Create temporary node for multi-hop atomic relationship 'people_person_places_lived, people_place_lived_location'
CREATE (tempNode:Node {name: "temp"})

// Connect Prith Banerjee to the country he lived in with 'people_person_places_lived, people_place_lived_location' relationship
CREATE (prith)-[:people_person_places_lived]->(tempNode)
CREATE (tempNode)-[:people_place_lived_location]->(usa)

// Connect USA to its currency
CREATE (usa)-[:location_country_currency_used]->(usd)

<step 3> {Completed Knowledge Graph}: 
// Create Prith Banerjee node
CREATE (prith:Person {name: "Prith Banerjee"})

// Create United States node
CREATE (usa:Country {name: "United States"})

// Create a node for the currency
CREATE (usd:Currency {name: "United States Dollar"})

// Create temporary node for multi-hop atomic relationship 'people_person_places_lived, people_place_lived_location'
CREATE (tempNode:Node {name: "temp"})

// Connect Prith Banerjee to the country he lived in with 'people_person_places_lived, people_place_lived_location' relationship
CREATE (prith)-[:people_person_places_lived]->(tempNode)
CREATE (tempNode)-[:people_place_lived_location]->(usa)

// Connect USA to its currency
CREATE (usa)-[:location_country_currency_used]->(usd)

// Adding another country that uses the USD for context
CREATE (ecuador:Country {name: "Ecuador"})
CREATE (ecuadorUsd:Currency {name: "United States Dollar"})

// Connect Ecuador to its currency
CREATE (ecuador)-[:location_country_currency_used]->(ecuadorUsd)

// Connect Prith Banerjee to Ecuador
CREATE (prith)-[:people_person_places_lived]->(tempNode)
CREATE (tempNode)-[:people_place_lived_location]->(ecuador)

<step 4> {Relation Paths}: 
path1: [(prith)-[:people_person_places_lived]->(tempNode)-[:people_place_lived_location]->(usa)-[:location_country_currency_used]->(usd)]
path2: [(prith)-[:people_person_places_lived]->(tempNode)-[:people_place_lived_location]->(ecuador)-[:location_country_currency_used]->(ecuadorUsd)]


[Example 8]:
{Question}: who wrote the novel 1984?
{Possible Atomic Relationships}: ['literary_author_written_work', 'people_written_works', 'literary_written_works_author']

<step 1> {Knowledge Planning}:  
To answer this question, we need to gather the following details:
- Information about the author of *1984*.
- The work that the author is associated with.

<step 2> {Knowledge Graph}:  
// Create node for the novel 1984
CREATE (novel:Book {title: "1984"})  
  
// Create node for the author George Orwell  
CREATE (orwell:Person {name: "George Orwell"})  
  
// Create relationship connecting George Orwell to 1984
CREATE (orwell)-[:literary_author_written_work]->(novel)

<step 3> {Completed Knowledge Graph}:  
// Create node for the novel 1984 
CREATE (novel:Book {title: "1984"})  
  
// Create node for the author George Orwell  
CREATE (orwell:Person {name: "George Orwell"})  
  
// Create relationship connecting George Orwell to 1984  
CREATE (orwell)-[:literary_author_written_work]->(novel)  

<step 4> {Relation Paths}:  
path1: [(orwell)-[:literary_author_written_work]->(novel)]


[Task]: 
{Question}: %s
{Possible Atomic Relationships}: %s'''


CONSTRUCT_AUXILIARY_INFORMATION_CHINESE_PROMPT = '''
[任务描述]: 
你需要按以下step回答{Question}: 
<step 1> 找出解决{Question}需要哪些{Knowledge Planing}。
<step 2> 使用{Possible Atomic Relationships}中的原子关系，基于你自己的知识，严格填写{Knowledge Planing}，尽可能完整地构建{Knowledge Graph}（使用Cypher语句）。
<step 3> 严格完成{Knowledge Graph}，构建{Completed Knowledge Graph}，以包括更多详细的推理Path来解决{Question}。
<step 4> 基于{Completed Knowledge Graph}，生成有效的{Relation Paths}，帮助回答{Question}。


[示例 1]: 
{Question}: Justin Bieber 的弟弟叫什么名字？
{Possible Atomic Relationships}: ['人物_人物_兄弟姐妹, 人物_兄弟姐妹关系_兄弟姐妹', '皇室_皇室血统_继承顺序', '人物_姓氏_拥有此姓氏的人']

<step 1> {Knowledge Planing}: 
为了解答 "Justin Bieber 的弟弟叫什么名字？" 这个问题，我们需要收集 Justin Bieber 的家庭成员信息，特别是他的兄弟姐妹。以下是我们在知识图谱中需要包含的关键信息：
- 介绍：
  - 简要介绍 Justin Bieber 及其家族背景。
- 兄弟姐妹：
  - 列出 Justin Bieber 的兄弟姐妹的名字。

<step 2> {Knowledge Graph}: 
// 创建 Justin Bieber 节点
CREATE (justin:Person {name: "Justin Bieber"})

// 创建 Justin Bieber 的弟弟节点
CREATE (justinBrother:Person {name: "Jaxon Bieber"})

// 创建用于多跳原子关系的临时节点 '人物_人物_兄弟姐妹, 人物_兄弟姐妹关系_兄弟姐妹'
CREATE (tempNode:Node {name: "temp"})

// 将 Justin Bieber 和他的弟弟通过 '人物_人物_兄弟姐妹, 人物_兄弟姐妹关系_兄弟姐妹' 关系连接
CREATE (justin)-[:人物_人物_兄弟姐妹]->(tempNode)
CREATE (tempNode)-[:人物_兄弟姐妹关系_兄弟姐妹]->(justinBrother)

<step 3> {Completed Knowledge Graph}: 
// 创建 Justin Bieber 节点
CREATE (justin:Person {name: "Justin Bieber"})

// 创建 Justin Bieber 的家庭成员节点
CREATE (justinBrother:Person {name: "Jaxon Bieber"})
CREATE (justinFather:Person {name: "Jeremy Bieber"})
CREATE (justinMother:Person {name: "Pattie Mallette"})

// 创建用于多跳原子关系的临时节点 '人物_人物_兄弟姐妹, 人物_兄弟姐妹关系_兄弟姐妹'
CREATE (tempNode:Node {name: "temp"})

// 将 Justin Bieber 和他的弟弟通过 '人物_人物_兄弟姐妹, 人物_兄弟姐妹关系_兄弟姐妹' 关系连接
CREATE (justin)-[:人物_人物_兄弟姐妹]->(tempNode)
CREATE (tempNode)-[:人物_兄弟姐妹关系_兄弟姐妹]->(justinBrother)

// 添加 Justin Bieber 的父母关系以帮助寻找弟弟
CREATE (justin)-[:父母]->(justinFather)
CREATE (justinFather)-[:孩子]->(justin)
CREATE (justinBrother)-[:父母]->(justinFather)
CREATE (justinFather)-[:孩子]->(justinBrother)
CREATE (justin)-[:父母]->(justinMother)
CREATE (justinMother)-[:孩子]->(justin)
CREATE (justinBrother)-[:父母]->(justinMother)
CREATE (justinMother)-[:孩子]->(justinBrother)

<step 4> {Relation Paths}: 
Path1: [(justin)-[:人物_人物_兄弟姐妹]->(tempNode)-[:人物_兄弟姐妹关系_兄弟姐妹]->(justinBrother)]
Path2: [(justin)-[:父母]->(justinFather)-[:孩子]->(justinBrother)]
Path3: [(justin)-[:父母]->(justinMother)-[:孩子]->(justinBrother)]


[任务]: 
{Question}: %s
{Possible Atomic Relationships}: %s'''


QUESTION_DECOMPOSITION_PROMPT = '''
[Task description]:
Break the <question> down into <atomic questions> according to subject-predicate-object as a list. If the <question> is already atomic, it can not be split. Finally, replace the subject of each <atomic question> by <topic entity> to generate <abstract atomic questions>.

[Example 1]: 
<question>: which kennedy died first?
<atomic questions>: ['which kennedy died first?']
<abstract atomic questions>: ['which <topic entity> died first?']

[Example 2]: 
<question>: what is nina dobrev nationality?
<atomic question>: ['what is nina dobrev nationality?']
<abstract atomic questions>: ['what is <topic entity> nationality?']

[Example 3]: 
<question>: Who were the key figures in the French Revolution?
<atomic question>: ['Who were the key figures in the French Revolution?']
<abstract atomic questions>: ['Who were the key figures in the <topic entity>?']

[Example 4]: 
<question>: what were the release years the films starred by Jean Rochefort?
<atomic questions>: ['What films did Jean Rochefort star in?', 'What were the release years of these films?']
<abstract atomic questions>: ['What films did <topic entity> star in?', 'What were the release years of these films?']

[Example 5]: 
<question>: what genres are the movies written by [Gene Wilder] in?
<atomic questions>: ['What movies were written by Gene Wilder?', 'What genres do these movies belong to?']
<abstract atomic questions>: ['What movies were written by <topic entity>?', 'What genres do these movies belong to?']

[Example 6]: 
<questions>: what are the main languages in [Elissa Landi] starred movies?
<atomic questions>: ['What movies did Elissa Landi star in?', 'What are the main languages of these movies?']
<abstract atomic questions>: ['What movies did <topic entity> star in?', 'What are the main languages of these movies?']

[Task]: 
<questions>: %s
<atomic questions>: 
<abstract atomic questions>: '''

QUESTION_DECOMPOSITION_PROMPT_CWQ = '''
[Task description]:
Break the <question> down into <atomic questions> according to subject-predicate-object as a list. If the <question> is already atomic, it can not be split. Finally, replace the subject of each <atomic question> by <topic entity> to generate <abstract atomic questions>.

[Example 1]: 
<question>: which kennedy died first?
<atomic questions>: ['which kennedy died first?']
<abstract atomic questions>: ['which <topic entity> died first?']

[Example 2]: 
<question>: Who were the key figures in the French Revolution?
<atomic question>: ['Who were the key figures in the French Revolution?']
<abstract atomic questions>: ['Who were the key figures in the <topic entity>?']

[Example 3]: 
<question>: What movie with film character named Mr. Woodson did Tupac star in?
<atomic questions>: ['What movies did Tupac star in?', 'Which of these movies with the character named Mr. Woodson?']
<abstract atomic questions>: ['What movies did <topic entity> star in?', 'Which of these movies with the character named <topic entity>?']

[Example 4]: 
<question>: What country sharing borders with Spain does the Set\u00c3\u00babal District belong to?
<atomic questions>: ['What country sharing borders with Spain?', 'What country does the Set\u00c3\u00babal District belong to?']
<abstract atomic questions>: ['What country sharing borders with <topic entity>?', 'What country does <topic entity> belong to?']

[Example 5]: 
<question>: What is the name of the president of the geographic location where Nicolas Sarkozy was appointed to a governmental position?
<atomic questions>: ['Where was Nicolas Sarkozy appointed to a governmental position?', 'What is the name of the president of this geographic location?']
<abstract atomic questions>: ['Where was <topic entity> appointed to a governmental position?', 'What is the name of the president of this geographic location?']

[Example 6]: 
<question>: What educational institution with mascot named Yale University Handsome Dan did James Franco go to?
<atomic questions>: ['What educational institution did James Franco go to?', 'Which of them with mascot named Yale University Handsome Dan?']
<abstract atomic questions>: ['What educational institution did <topic entity> go to?', 'Which of them with mascot named <topic entity>?']

[Example 7]: 
<question>: What language is spoken in the location that appointed Margrethe II of Denmark to a governmental position?
<atomic questions>: ['What is the location where Margrethe II of Denmark was appointed to a governmental position?', 'What language is spoken in this location?']
<abstract atomic questions>: ['What is the location where <topic entity> was appointed to a governmental position?', 'What language is spoken in this location?']

[Example 8]: 
<question>: What country did Prith Banerjee in that uses United States Dollar?
<atomic questions>: ['What country did Prith Banerjee live in?', 'Which of those countries use United States Dollar?']
<abstract atomic questions>: ['What country did <topic entity> live in?', 'Which of those countries use <topic entity>?']


[Task]: 
<questions>: %s
<atomic questions>: 
<abstract atomic questions>: '''


QUESTION_DECOMPOSITION_CHINESE_NEW_PROMPT = '''
[任务描述]: 
将<question>按照主语-谓语-宾语的顺序分解成<atomic questions>。如果<question>已经是原子性的，则不能拆分它。最后，用<topic entity>替换每个<atomic questions>的主题，生成<abstract atomic questions>。

[例1]: 
<question>: 医生您好，我最近感觉喉咙里有异物感，去检查说是扁桃体结石。之前已经发生过两次，但这次不敢去医院，想知道有没有在家可以处理的方法？（男，42岁）平时有慢性扁桃体炎，喉咙总是发炎，偶尔发烧。
<atomic questions>: ['扁桃体结石有没有在家可以处理的方法?']
<abstract atomic questions>: ['<topic entity>有没有在家可以处理的方法?']


[例2]: 
<question>: 前天在房间里用了煤炭取暖，可能是一氧化碳中毒，凌晨感觉难受，出去后晕倒，醒来后整天头疼。朋友提醒说一氧化碳中毒有潜伏期，我该去医院做进一步检查吗？会不会有后遗症？（男，34岁）
<atomic questions>: ['一氧化碳中毒要不要去医院做进一步检查？', '一氧化碳中毒会不会有后遗症？']
<abstract atomic questions>: ['<topic entity>要不要去医院做进一步检查？', '<topic entity>会不会有后遗症？']


[例3]: 
<question>: 您好，我爸爸得了肝癌去世，外婆也是因胃癌去世。我想知道这些癌症会遗传给我吗？我今年24岁，体检没有乙肝病毒携带，平时身体健康，是否可以忽略遗传因素？（女，24岁）
<atomic questions>: ['肝癌和胃癌会遗传吗？', '我可以忽略遗传因素吗？']
<abstract atomic questions>: ['<topic entity>会遗传吗？', '<topic entity>可以忽略遗传因素吗？']


[例4]: 
<question>: 医生您好，我家宝宝10个月大，最近发烧到38.5度，退烧后全身出了一些红疹，宝宝非常烦躁，不吃东西。我担心是幼儿急疹，请问有什么护理建议？出疹期间可以洗澡吗？（女，10个月）
<atomic questions>: ['幼儿急疹有什么护理建议？', '宝宝出疹期间可以洗澡吗？']
<abstract atomic questions>: ['<topic entity>有什么护理建议？', '<topic entity>可以洗澡吗？']


[例5]: 
<question>: 大夫，最近我的鼻子一侧经常疼痛，连带眼眶和太阳穴也疼，有时咳嗽或打喷嚏会加重。这种情况已经持续了几个月，CT报告显示有颌窦炎。会不会有其他严重的疾病，比如肿瘤？（女，56岁）
<atomic questions>: ['颌窦炎会不会引发其他严重疾病？']
<abstract atomic questions>: ['<topic entity>会不会引发其他严重疾病？']


[例6]: 
<question>: 医生，我有慢性荨麻疹，时不时会发作，主要在出汗或者休息不好时。我下个月要注射新冠疫苗，但担心荨麻疹会加重或者引发严重的过敏反应，想知道是否可以安全接种？（男，25岁）
<atomic questions>: ['慢性荨麻疹患者注射新冠疫苗会引发严重过敏反应吗？']
<abstract atomic questions>: ['<topic entity>会引发严重过敏反应吗？']


[任务]: 
<question>: %s
<atomic questions>: 
<abstract atomic questions>:
'''

GEN_ANSWER_BY_REASONING_PATH_PTOMPT_1003 = '''
[Task description]: 
Given a {Question} and the associated retrieved {Knowledge}, you are asked to answer the {Question} with these retrieved knowledge and keep all the candidate answers. And then give the final {Answer} and return all the possible answers as a list. And explain why. 

[Example 1]: 
{Question}: what is the name of justin bieber brother?

{Knowledge}: 
Justin Bieber, people_person_sibling_s, m.0gxnnwc, people_sibling_relationship_sibling, Jazmyn Bieber
Justin Bieber, people_person_sibling_s, m.0gxnnwp, people_sibling_relationship_sibling, Jaxon Bieber
Jaxon Bieber, people_person_sibling_s, m.0gxnnwp, people_sibling_relationship_sibling, Justin Bieber
Jaxon Bieber, people_person_parents, Jeremy Bieber, people_person_children, Justin Bieber
Jaxon Bieber, people_person_parents, Jeremy Bieber, people_person_children, Jazmyn Bieber
Justin Bieber, people_person_parents, Jeremy Bieber, people_person_children, Jazmyn Bieber
Justin Bieber, people_person_parents, Jeremy Bieber, people_person_children, Justin Bieber
Justin Bieber, people_person_parents, Jeremy Bieber, people_person_children, Jaxon Bieber
Justin Bieber, people_person_sibling_s, m.0gxnnwc
Justin Bieber, people_person_sibling_s, m.0gxnnwp
m.0gxnnwp, people_sibling_relationship_sibling, Jaxon Bieber
m.0gxnnwc, people_sibling_relationship_sibling, Jazmyn Bieber
Justin Bieber, people_person_parents, Jeremy Bieber
Jeremy Bieber, people_person_children, Jaxon Bieber
Jaxon Bieber, people_person_sibling_s, m.0gxnnwp
Jaxon Bieber, people_person_parents, Jeremy Bieber
Jazmyn Bieber, people_person_parents, Jeremy Bieber
Jeremy Bieber, people_person_children, Jazmyn Bieber
Jeremy Bieber, people_person_children, Justin Bieber

{Answer}: ['Jaxon Bieber']

{Explanation}: The knowledge clearly indicates that Justin Bieber has two siblings, Jazmyn and Jaxon. Since Jazmyn is his sister, Jaxon is his brother, making Jaxon Bieber the correct answer.


[Example 2]: 
{Question}: which countries border the us?

{Knowledge}: 
North America, base_locations_continents_countries_within, Canada
North America, base_locations_continents_countries_within, Mexico
Edmonton, base_biblioness_bibs_location_country, Canada
Prince Edward Island, base_biblioness_bibs_location_country, Canada
Mississauga, base_biblioness_bibs_location_country, Canada
Hipódromo, location_administrative_division_country, Mexico
Sinaloa, location_administrative_division_country, Mexico
Guanajuato, location_administrative_division_country, Mexico
Esperanza, location_administrative_division_country, Mexico
Cozumel, location_administrative_division_country, Mexico
Mexico, location_location_containedby, United States of America, location_location_partially_contains, Boundary Butte, location_location_partially_containedby, Canada
Mexico, location_location_containedby, United States of America, location_location_partially_contains, Great Lakes, location_location_partially_containedby, Canada

{Answer}: ['Canada', 'Mexico']

{Explanation}: The retrieved knowledge lists two countries as being within North America that are associated with bordering the US: Canada and Mexico. Therefore, based on the knwloedge, we can conclude that the countries bordering the US are Canada and Mexico.


[Example 3]: 
{Question}: what books did beverly cleary right?

{Knowledge}: 
m.0w1p363, people_marriage_spouse, Bob Huggins
Denise DiNovi, film_producer_film, Ramona and Beezus
m.0z8vhl7, award_award_nomination_nominated_for, Ramona and Beezus
West Virginia Mountaineers mens basketball, basketball_basketball_team_head_coach, Bob Huggins
m.05kgzmn, basketball_basketball_historical_coach_position_coach, Bob Huggins
Beverly, tv_tv_character_appeared_in_tv_episodes, m.09nzlhv
Beverly, common_topic_notable_for, g.1255wtkh2
Beverly, location_location_people_born_here, Mike Castle
Beverly, common_topic_article, m.038khp
Bullpen Rib House, common_topic_subjects, Ribs
Beverly, common_topic_notable_types, Fictional Character
m.0666mpt, film_performance_film, Ramona and Beezus
San Diego County, location_us_county_hud_county_place, Ramona
Ramona, location_hud_county_place_place, Ramona

{Answer}: ['Ramona the Brave', 'Dear Mr. Henshaw', 'Ralph S. Mouse', 'Beezus and Ramona', 'Two times the fun', 'Henry Huggins']

{Explanation}: The retrieved knowledge does not directly mention the books written by Beverly Cleary. However, based on general knowledge, Beverly Cleary is known for writing children's books, such as the Ramona the Brave, Dear Mr. Henshaw, Ralph S. Mouse, Beezus and Ramona, Two times the fun, Henry Huggins and so on. Therefore, we can conclude that Beverly Cleary wrote books such as Ramona the Brave, Dear Mr. Henshaw, Ralph S. Mouse, Beezus and Ramona, Two times the fun, Henry Huggins.


[Example 4]: 
{Question}: what do people in the czech republic speak?

{Knowledge graph triples}: 
Late Medieval Germany, location_country_official_language, Slavic language
Czech Republic, location_country_languages_spoken, Romani language
Czech Republic, location_country_languages_spoken, Rusyn Language
Czech Republic, location_country_languages_spoken, Bulgarian Language
Czech Republic, location_country_languages_spoken, Czech Language
Czech Republic, location_country_languages_spoken, Hungarian language
Czech Republic, location_country_languages_spoken, Greek Language
Czech Republic, location_country_languages_spoken, German Language
Czech Republic, location_country_languages_spoken, Polish Language
Czech Republic, location_country_languages_spoken, Russian Language
Czech Republic, location_country_languages_spoken, Slovak Language
Czech Republic, location_country_languages_spoken, Ukrainian Language
Czech Republic, location_country_languages_spoken, Serbian language
Czech Republic, location_country_languages_spoken, Croatian language
Indo-European languages, common_topic_image, Indo-European languages
Czech Language, base_rosetta_languoid_local_name, Czech
Czech Republic, government_political_district_representatives, m.0h2zs1k

{Answer}: ['Czech Language', 'Romani language', 'Rusyn Language', 'Bulgarian Language', 'Hungarian language', 'Greek Language', 'German Language', 'Polish Language', 'Russian Language', 'Slovak Language', 'Ukrainian Language', 'Serbian language', 'Croatian language']

{Explanation}: The retrieved knowledge lists many languages spoken in the Czech Republic. The languages mentioned are Czech, Romani, Rusyn, Bulgarian, Hungarian, Greek, German, Polish, Russian, Slovak, Ukrainian, Serbian, and Croatian are also spoken in the country. Therefore, people in the czech republic speak Czech Language, Romani language, Rusyn Language, Bulgarian Language, Hungarian language, Greek Language, German Language, Polish Language, Russian Language, Slovak Language, Ukrainian Language, Serbian language and Croatian language.


[Example 5]: 
{Question}: what year did pete rose play?

{Knowledge}: 
Pete Rose, baseball_baseball_player_batting_stats, m.027pwzc, baseball_batting_statistics_season, 1985 Major League Baseball season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s1h, baseball_batting_statistics_season, 1973 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s1n, baseball_batting_statistics_season, 1964 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s1t, baseball_batting_statistics_season, 1982 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s28, baseball_batting_statistics_season, 1966 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s2f, baseball_batting_statistics_season, 1971 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s31, baseball_batting_statistics_season, 1963 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s36, baseball_batting_statistics_season, 1978 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s3j, baseball_batting_statistics_season, 1970 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s3p, baseball_batting_statistics_season, 1979 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s4s, baseball_batting_statistics_season, 1967 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s52, baseball_batting_statistics_season, 1984 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s3j, baseball_batting_statistics_season, 1970 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s5k, baseball.batting_statistics.season, 1983 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s5q, baseball.batting_statistics.season, 1976 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s6b, baseball.batting_statistics.season, 1972 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s78, baseball.batting_statistics.season, 1981 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s7r, baseball.batting_statistics.season, 1975 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s81, baseball.batting_statistics.season, 1977 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s8c, baseball.batting_statistics.season, 1980 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7s9g, baseball.batting_statistics.season, 1974 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7sbd, baseball.batting_statistics.season, 1968 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7sbq, baseball.batting_statistics.season, 1986 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.02h7sd3, baseball.batting_statistics.season, 1969 Major League Baseball Season
Pete Rose, baseball_baseball_player_batting_stats, m.05kcgsf, baseball.batting_statistics.season, 1965 Major League Baseball Season

{Answer}: ['1963 Major League Baseball Season', '1964 Major League Baseball Season', '1965 Major League Baseball Season', '1966 Major League Baseball Season', '1967 Major League Baseball Season', '1968 Major League Baseball Season', '1969 Major League Baseball Season', '1970 Major League Baseball Season', '1971 Major League Baseball Season', '1972 Major League Baseball Season', '1973 Major League Baseball Season', '1974 Major League Baseball Season', '1975 Major League Baseball Season', '1976 Major League Baseball Season', '1977 Major League Baseball Season', '1978 Major League Baseball Season', '1979 Major League Baseball Season', '1980 Major League Baseball Season', '1981 Major League Baseball Season', '1982 Major League Baseball Season', '1983 Major League Baseball Season', '1984 Major League Baseball Season', '1985 Major League Baseball Season', '1986 Major League Baseball Season']

{Explanation}: The retrieved knowledge provides various seasons Pete Rose played in Major League Baseball, spanning from 1963 to 1986. Specific years in which Pete Rose played include: 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, and 1986.


[Example 6]: 
{Question}: What is the traditional Japanese art of paper folding called?

{Knowledge}: 

{Answer}: ['Origami']

{Explanation}: The question is asking about the traditional Japanese art of paper folding, so we need to identify the name of this art form. Since the provided knowledge are empty, there is no direct information available to answer the question. Based on general knowledge, we know that the traditional Japanese art of paper folding is called Origami. The correct answer, therefore, is Origami.


[Task]: 
{Question}: %s
{Knowledge}: 
%s
'''

GEN_ANSWER_BY_REASONING_PATH_PTOMPT_MED = '''You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. 
Patient input: %s 
You have some medical knowledge information in the following: %s 
The final answer consists of three parts: 
1.What disease does the patient have? If it is not possible to determine from the MEDICAL background knowledge given above what disease the patient is suffering from then this section can be left unanswered or the patient can be referred for tests to determine what disease he may have. 
2.What tests should patient take to confirm the diagnosis? 
3.What recommended medications can cure the disease? Think step by step. 
Output: The answer includes disease and tests and recommended medications. 

There is an output sample: 
Output: Based on your symptoms, it sounds like you may have acute pancreatitis. To confirm this, we will need to run a series of medical tests. We will start with a blood test and a complete blood count (CBC), as well as a radiographic imaging procedure to determine the extent of the pancreatitis. We may also need to provide intravenous fluid replacement and perform kidney function tests and glucose level measurements. Additionally, a urinalysis will be necessary to check for any kidney damage.
'''

GEN_ANSWER_BY_REASONING_PATH_PTOMPT_MED_CHINESE = '''你是一名优秀的人工智能医生，你可以根据对话中的症状诊断疾病并推荐药物。
患者输入: %s
以下是您的一些医学知识信息: %s
最终答案由三部分组成: 
1.病人得了什么病?如果无法从上述医学背景知识中确定患者患的是什么疾病，则可以不回答这一部分，或者可以将患者转介进行检查，以确定他可能患有什么疾病。
2.患者需要做哪些检查来确诊?
3.推荐哪些药物可以治愈这种疾病?一步一步地思考。
Output: 答案包括疾病和测试以及推荐的药物。

一个输出示例:
Output: 根据你的症状，听起来你可能得了急性胰腺炎。为了证实这一点，我们需要进行一系列的医学测试。我们将从血液检查和全血细胞计数(CBC)以及放射成像程序开始，以确定胰腺炎的程度。我们可能还需要提供静脉补液，进行肾功能测试和血糖水平测量。此外，还需要进行尿液分析以检查肾脏是否受损。'''
