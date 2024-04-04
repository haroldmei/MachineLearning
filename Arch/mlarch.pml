@startuml
process Preproccessor
process FE[
    Feature engineering
    ---
    ticket:
    ---
    user:
    ---
    interaction:
]


process Modeller
process Retriever
process Ranker
process Indexer
process ContextRetriever
process ContextAgg

database User
database Ticket
database ut_interaction
database indexedEmb

collections rankedItem

actor endUser

User --> Preproccessor
Ticket --> Preproccessor
ut_interaction --> Preproccessor
Preproccessor --> FE

FE --> Modeller

Modeller --> Indexer        :emb
Indexer --> indexedEmb

endUser --> ContextRetriever
ut_interaction --> ContextRetriever

ContextRetriever --> ContextAgg     :list of tickets
ContextAgg --> Retriever    :aggregrated User embedding
indexedEmb --> Retriever

Retriever --> Ranker        : resulting items to be filtered by team
Ranker --> rankedItem

@enduml