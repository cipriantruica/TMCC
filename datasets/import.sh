#!/bin/bash
#The parameter is the path
PATH_IMPORT=$1

mongoimport --db ConferenceDB --collection documents --file "$PATH_IMPORT"ConferenceDB_documents.json
mongoimport --db ConferenceDB --collection vocabulary --file "$PATH_IMPORT"ConferenceDB_vocabulary.json
mongoimport --db NewsArticlesDB --collection documents --file "$PATH_IMPORT"NewsArticlesDB_documents.json
mongoimport --db NewsArticlesDB --collection vocabulary --file "$PATH_IMPORT"NewsArticlesDB_vocabulary.json
mongoimport --db ArxivDB --collection documents --file "$PATH_IMPORT"ArxivDB_documents.json
mongoimport --db ArxivDB --collection vocabulary --file "$PATH_IMPORT"ArxivDB_vocabulary.json 