package com.anon.rag.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public final class SearchResult {

    private final String searchItemId;
    private final List<CodeSnippet> scoredCodeSnippets = new ArrayList<>();

    public SearchResult(String searchItemId, Collection<CodeSnippet> codeSnippets) {
        this.searchItemId = searchItemId;
        if (codeSnippets.stream().anyMatch(x -> x.getMetaData("score") == null)) {
            throw new RuntimeException("CodeSnippet must contain the score");
        }
        this.scoredCodeSnippets.addAll(codeSnippets);
    }

    

    @Override
    public String toString() {
        return "{Search-Id:" + searchItemId + "}{Retrieved-Items:" + scoredCodeSnippets + "}";
    }

    public int size() {
        return scoredCodeSnippets.size();
    }



    public String getSearchItemId() {
        return searchItemId;
    }



    public List<CodeSnippet> getScoredCodeSnippets() {
        return scoredCodeSnippets;
    }

}
