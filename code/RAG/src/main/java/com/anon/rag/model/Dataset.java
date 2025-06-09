package com.anon.rag.model;

import java.util.HashMap;

public class Dataset {
    
    private HashMap<String, CodeSnippet> fileHeaders = new HashMap<>();

    public Dataset() {
    }

    public static Dataset merge(Dataset dataset1, Dataset dataset2) {
        for (var entry : dataset2.getDataset().entrySet()) {
            dataset1.addPair(entry.getKey(), entry.getValue());
        }
        return dataset1;
    }

    public void addPair(String id, String header, String full) {
        int bodyIndex = full.indexOf("{");
        if (bodyIndex > 0 ) {
            addPair(id, new CodeSnippet(id ,header, full.substring(bodyIndex).trim()));
        }
    }

    public void addPair(String id, CodeSnippet snippet) {
        // snippet.addMetaData(INDEX_STRING, id);
        fileHeaders.put(id, snippet);
    }

    public HashMap<String, CodeSnippet> getDataset() {
        return fileHeaders;
    }

    public int getSize() {
        return fileHeaders.size();
    }
}
