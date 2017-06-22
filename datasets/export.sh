#!/bin/bash
#The parameter is the path
PATH_EXPORT=$1

mongoexport --db ConferenceDB --collection documents --out "$PATH_EXPORT"ConferenceDB_documents.json
mongoexport --db ConferenceDB --collection vocabulary --out "$PATH_EXPORT"ConferenceDB_vocabulary.json
mongoexport --db NewsArticlesDB --collection documents --out "$PATH_EXPORT"NewsArticlesDB_documents.json
mongoexport --db NewsArticlesDB --collection vocabulary --out "$PATH_EXPORT"NewsArticlesDB_vocabulary.json
mongoexport --db ArxivDB --collection documents --out "$PATH_EXPORT"ArxivDB_documents.json
mongoexport --db ArxivDB --collection vocabulary --out "$PATH_EXPORT"ArxivDB_vocabulary.json

