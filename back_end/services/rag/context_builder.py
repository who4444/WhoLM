def build_doc_contect(hits):
    context_parts = []
    for i, hit in enumerate(hits):
        text = hit.payload.get("text", "")
        meta = hit.payload.get("metadata", {})
        filename = meta.get("filename", "Unknown File")
        page = meta.get("page_number", "?")

        citation_id = i + 1

        formatted_chunk = (
            f"[{citation_id}] (Source: {filename}, Page: {page}):\n"
            f"{text}\n"
        )
        context_parts.append(formatted_chunk)

    return "\n---\n".join(context_parts)

def build_video_context(search_hits):
    context_parts = []
    
    for hit in search_hits:
        text_content = hit.payload.get("text_context") or hit.payload.get("transcript")
        start = hit.payload.get("start_time", 0)
        end = hit.payload.get("end_time", 0)

        data_type = "Visual Description" if hit.payload.get("visual_context") else "Transcript"
        
        # Helper lambda for formatting
        fmt_time = lambda s: f"{int(s)//60:02d}:{int(s)%60:02d}"
        time_str = f"[{fmt_time(start)} - {fmt_time(end)}]"
        
        # Format: "[00:45 - 00:50] (Type): Text..."
        formatted_chunk = (
            f"{time_str} ({data_type}):\n"
            f"{text_content}\n"
        )
        context_parts.append(formatted_chunk)

    return "\n---\n".join(context_parts)