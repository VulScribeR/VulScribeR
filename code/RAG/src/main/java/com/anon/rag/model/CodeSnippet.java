package com.anon.rag.model;

import java.util.HashMap;
import java.util.Objects;
import java.util.Optional;

public class CodeSnippet implements Cloneable {

    public final String id;
    private final String functionHeader;
    private final String body;
    private transient HashMap<String, String> metaData = new HashMap<>();

    public CodeSnippet(String id, String functionHeader, String snippet) {
        this.functionHeader = functionHeader;
        this.body = snippet;
        this.id = id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.id);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj != null && obj instanceof CodeSnippet cs) {
            return cs.id.equals(this.id);
        }
        return false;
    }

    private CodeSnippet setMetaData(HashMap<String, String> metaData) {
        this.metaData = metaData;
        return this;
    }

    @Override
    public CodeSnippet clone() throws CloneNotSupportedException {
        return ((CodeSnippet) super.clone()).setMetaData(metaData);
    }

    public String getMetaData(String key) {
        return metaData.get(key);
    }

    public String getScore(){
        return this.metaData.get("score");
    }

    public String getClusterIndex(){
        return Optional.ofNullable(this.metaData.get("clusterIndex")).orElse("-1");
    }

    public String addMetaData(String key, String value) {
        return metaData.put(key, value);
    }

    public String getHeader() {
        return functionHeader;
    }

    // public static String getAnonymizedHeader(String header){
    //     // TODO
    //     throw new UnsupportedOperationException();
    // }

    // public static String getAnonymizedBody(String body){ // TODO
    //     throw new UnsupportedOperationException();
    // }

    public String getBody() {
        return body;
    }

    @Override
    public String toString() {
        StringBuilder metaBuilder = new StringBuilder();
        metaBuilder.append("<META>");
        for (String key : this.metaData.keySet()) {
            metaBuilder.append(key)
                    .append(":")
                    .append(this.metaData.get(key))
                    .append(",");
        }
        metaBuilder.append("</META>")
                .append(this.functionHeader).append("\n").append(this.body);
        return metaBuilder.toString();
    }

    public String getId() {
        return this.id;
    }

    public String toStringV1() {
        String base = "{header:" + this.getHeader() + "}\n" + "{body:" + this.body + "}";
        StringBuilder metaBuilder = new StringBuilder();
        for (String key : this.metaData.keySet()) {
            metaBuilder.append("{")
                    .append(key).append(":")
                    .append(this.metaData.get(key))
                    .append("}");
        }
        return base + metaBuilder.toString();
    }

    public String getFunction(){
        return this.functionHeader + "\n" + this.body;
    }
}
