#
# Script to run the OHDSI Data Quality Tool on a generated dataset
# As of 2025-11-11 this will report errors for several fields in 
# VISIT_OCCURENCE and VISIT_DETAIL because some columns were renamed
# in CDM 5.4 and the DQD does not yet account for that
#

library(duckdb)
library(DBI)
library(DataQualityDashboard)

inputFile <- "dataset.db"

# Ensure an empty results schema exists before starting
con <- dbConnect(duckdb(), dbdir = inputFile, read_only = FALSE)
try({
    dbExecute(con, "drop schema results cascade")
})
try({
    dbExecute(con, "create schema results")
})
rm(con)

connectionDetails <- createConnectionDetails(dbms = "duckdb", 
                                             server = inputFile)

cdmDatabaseSchema <- "main" # the fully qualified database schema name of the CDM
resultsDatabaseSchema <- "results" # the fully qualified database schema name of the results schema (that you can write to)
cdmSourceName <- "Synthetic" # a human readable name for your CDM source
cdmVersion <- "5.4" # the CDM version you are targetting. Currently supports 5.2, 5.3, and 5.4

outputFolder <- "output"
outputFile <- "dqd_output.json"

executeDqChecks(
    connectionDetails = connectionDetails, 
    cdmDatabaseSchema = cdmDatabaseSchema, 
    resultsDatabaseSchema = resultsDatabaseSchema,
    cdmSourceName = cdmSourceName,
    outputFolder = outputFolder,
    outputFile = outputFile
)

outputPath <- paste0(outputFolder, '/', outputFile)
outputJson <- readChar(outputPath, file.info(outputPath)$size)
viewDqDashboard(outputJson)
