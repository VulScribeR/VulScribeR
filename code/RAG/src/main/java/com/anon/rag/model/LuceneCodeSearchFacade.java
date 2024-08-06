package com.anon.rag.model;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queries.mlt.MoreLikeThis;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

import com.anon.rag.model.CodeIndexer.Utils;

/**
 *
 * @author anonymous author
 */
public class LuceneCodeSearchFacade {

    private final Directory indexDirectory;
    private final Analyzer analyzer;
    private final CodeIndexer indexer;

    /**
     *
     * @throws IOException
     */
    public LuceneCodeSearchFacade() throws IOException {
        this.indexDirectory = new ByteBuffersDirectory(); // do on disk
        this.analyzer = new StandardAnalyzer();
        this.indexer = new CodeIndexer(indexDirectory, analyzer);
    }

    public LuceneCodeSearchFacade index(Collection<CodeSnippet> codeSnippets) throws IOException {
        indexer.index(codeSnippets);
        return this;
    }

    public SearchResult findSimilarHeader(String originalId, String header) {
        return findSimilarHeader(originalId, header, 5);
    }

    public SearchResult findHeader2Code(String originalId, String header) {
        return findHeader2Code(originalId, header, 5);
    }

    public SearchResult findSimilarFunction(String originalId, String function) {
        return findSimilarFunction(originalId, function, 5);
    }


    public SearchResult findSimilarHeader(String originalId, String header, int topN) {
        if (this.indexer.isIndexed() == false) {
            throw new RuntimeException("Not indexed yet!");
        }

        List<CodeSnippet> results = new ArrayList<>();
        try (IndexReader reader = DirectoryReader.open(this.indexDirectory)) {
            IndexSearcher indexSearcher = new IndexSearcher(reader);
            MoreLikeThis mlt = new MoreLikeThis(reader);
            mlt.setAnalyzer(analyzer);
            mlt.setMinTermFreq(0);
            mlt.setMinDocFreq(0);
            mlt.setFieldNames(new String[]{"id", "header", "body", "function"});
            // todo Reader -> query -> topdocs -> scoreDocs -> retreive -> read (do this for all + write to jsonl)
            Reader headerReader = new StringReader(header);
            Query query = mlt.like("header", headerReader);
            TopDocs topDocs = indexSearcher.search(query, topN);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                float score = scoreDoc.score;
                // if (score >= 8) {
                // Explanation explanation = indexSearcher.explain(query, scoreDoc.doc);
                // System.out.println(explanation);
                // }
                Document similarItem = indexSearcher.doc(scoreDoc.doc);
                CodeSnippet codeSnippet = Utils.convertDocumentToCodeSnippet(similarItem);
                codeSnippet.addMetaData("score", String.valueOf(score));
                results.add(codeSnippet);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return new SearchResult(originalId, results);
    }

    public SearchResult findSimilarFunction(String originalId, String function, int topN) {
        if (this.indexer.isIndexed() == false) {
            throw new RuntimeException("Not indexed yet!");
        }

        List<CodeSnippet> results = new ArrayList<>();
        try (IndexReader reader = DirectoryReader.open(this.indexDirectory)) {
            IndexSearcher indexSearcher = new IndexSearcher(reader);
            MoreLikeThis mlt = new MoreLikeThis(reader);
            mlt.setAnalyzer(analyzer);
            mlt.setMinTermFreq(0);
            mlt.setMinDocFreq(0);
            mlt.setFieldNames(new String[]{"id", "header", "body", "function"});
            // todo Reader -> query -> topdocs -> scoreDocs -> retreive -> read (do this for all + write to jsonl)
            Reader functionReader = new StringReader(function);
            Query query = mlt.like("function", functionReader);
            TopDocs topDocs = indexSearcher.search(query, topN);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                float score = scoreDoc.score;
                // if (score >= 8) {
                // Explanation explanation = indexSearcher.explain(query, scoreDoc.doc);
                // System.out.println(explanation);
                // }
                Document similarItem = indexSearcher.doc(scoreDoc.doc);
                CodeSnippet codeSnippet = Utils.convertDocumentToCodeSnippet(similarItem);
                codeSnippet.addMetaData("score", String.valueOf(score));
                results.add(codeSnippet);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return new SearchResult(originalId, results);
    }

    public SearchResult findHeader2Code(String originalId, String header, int topN) {
        if (this.indexer.isIndexed() == false) {
            throw new RuntimeException("Not indexed yet!");
        }

        List<CodeSnippet> results = new ArrayList<>();
        try (IndexReader reader = DirectoryReader.open(this.indexDirectory)) {
            IndexSearcher indexSearcher = new IndexSearcher(reader);
            MoreLikeThis mlt = new MoreLikeThis(reader);
            mlt.setAnalyzer(analyzer);
            mlt.setMinTermFreq(0);
            mlt.setMinDocFreq(0);
            mlt.setFieldNames(new String[]{"id", "header", "body", "function"});
            // todo Reader -> query -> topdocs -> scoreDocs -> retreive -> read (do this for all + write to jsonl)
            Reader headerReader = new StringReader(header);
            Query query = mlt.like("function", headerReader);
            TopDocs topDocs = indexSearcher.search(query, topN);
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                float score = scoreDoc.score;
                Document similarItem = indexSearcher.doc(scoreDoc.doc);
                CodeSnippet codeSnippet = Utils.convertDocumentToCodeSnippet(similarItem);
                codeSnippet.addMetaData("score", String.valueOf(score));
                results.add(codeSnippet);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        return new SearchResult(originalId, results);
    }
}

final class CodeIndexer {

    private final IndexWriter writer;
    private boolean indexed = false;

    protected CodeIndexer(Directory indexDirectory, Analyzer analyzer) throws IOException {
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig(analyzer);
        indexWriterConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        this.writer = new IndexWriter(indexDirectory, indexWriterConfig);
    }

    /**
     *
     * @param codeSnippets
     * @return
     * @throws IOException
     */
    public void index(Collection<CodeSnippet> codeSnippets) throws IOException {
        try (writer) {
            writer.commit();
            List<Document> documents = codeSnippets.stream()
                    .map(Utils::covertCodeSnippetToDocument)
                    .toList();
            writer.addDocuments(documents);
            writer.forceMerge(100, true);
        }
        this.indexed = true;
    }

    public Boolean isIndexed() {
        return this.indexed;
    }

    public static final class Utils {

        private Utils() {
        }

        public static CodeSnippet convertDocumentToCodeSnippet(Document document) {
            String id = document.get("id");
            String header = document.get("header");
            String body = document.get("body");
            return new CodeSnippet(id, header, body);
        }

        public static Document covertCodeSnippetToDocument(CodeSnippet snippet) {
            FieldType idType = new FieldType();

            idType.setIndexOptions(IndexOptions.DOCS);
            idType.setStored(true);
            idType.setStoreTermVectors(true); //TermVectors are needed for MoreLikeThis
            Field id = new Field("id", snippet.getId(), idType);

            FieldType headerType = new FieldType();
            headerType.setIndexOptions(IndexOptions.DOCS);
            headerType.setStored(true);
            headerType.setStoreTermVectors(true);
            Field header = new Field("header", snippet.getHeader(), headerType);

            FieldType bodyType = new FieldType();
            bodyType.setIndexOptions(IndexOptions.DOCS);
            bodyType.setStored(true);
            bodyType.setStoreTermVectors(true); //TermVectors are needed for MoreLikeThis
            Field body = new Field("body", snippet.getBody(), bodyType);


            FieldType functionType = new FieldType();
            functionType.setIndexOptions(IndexOptions.DOCS);
            functionType.setStored(true);
            functionType.setStoreTermVectors(true); //TermVectors are needed for MoreLikeThis
            Field function = new Field("function", snippet.getFunction(), functionType);

            Document document = new Document();
            // document.add(new StringField("id", snippet.getId(), Field.Store.YES));
            document.add(id);
            document.add(header);
            document.add(body);
            document.add(function);
            return document;
        }
    }

}
