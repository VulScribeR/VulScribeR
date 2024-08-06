package com.anon.rag.utils;

import java.util.function.BiFunction;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

import com.anon.rag.model.Dataset;

public class DatasetTest {
    @Test
    public void testReadDatasets() throws Exception{
        BiFunction<Dataset, String, Integer> checker = (Dataset ds, String str) -> {
            // System.out.println(ds.getDataset().get(str));
            System.out.println(ds.getSize());
            return ds.getSize();
        };
        assertEquals(11858, (Object) checker.apply(DatasetUtils.readCleanDevign(), "FFmpeg_0a41f47dc17b49acaff6fe469a6ab358986cc449_0_dv.c_nonvul.c_0.c")); // 12024 -> some code snippets are not functions -> 11858
        assertEquals(6610, (Object) checker.apply(DatasetUtils.readFormattedVulsBigVul(), "186765"));
        assertEquals(142125, (Object) checker.apply(DatasetUtils.readCleanBigvul(), "171392"));

        // System.out.println(DatasetUtils.readCleanBigvul().getDataset().get("171392").getFunction());
    }



    @Test
    public void testExtractMethodHeader() {
    
        var x = DatasetUtils.extractMethodHeader("""
            void RenderFrameImpl::JavaScriptExecuteRequestInIsolatedWorld(
            const base::string16& javascript,
            int32_t world_id,
            JavaScriptExecuteRequestInIsolatedWorldCallback callback) {
          TRACE_EVENT_INSTANT0("test_tracing",
                               "JavaScriptExecuteRequestInIsolatedWorld",
                               TRACE_EVENT_SCOPE_THREAD);

          if (world_id <= ISOLATED_WORLD_ID_GLOBAL ||
              world_id > ISOLATED_WORLD_ID_MAX) {
            NOTREACHED();
            std::move(callback).Run(base::Value());
            return;
          }
          v8::HandleScope handle_scope(v8::Isolate::GetCurrent());
          WebScriptSource script = WebScriptSource(WebString::FromUTF16(javascript));
          JavaScriptIsolatedWorldRequest* request = new JavaScriptIsolatedWorldRequest(
              weak_factory_.GetWeakPtr(), std::move(callback));
          frame_->RequestExecuteScriptInIsolatedWorld(
              world_id, &script, 1, false, WebLocalFrame::kSynchronous, request);
        }
        """);
        System.out.println(x);
        assertEquals("void RenderFrameImpl::JavaScriptExecuteRequestInIsolatedWorld(const base::string16& javascript, int32_t world_id, JavaScriptExecuteRequestInIsolatedWorldCallback callback)", x);
    }
}
