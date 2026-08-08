-- scholarly-metadata.lua
-- Converts JOSS-style authors/affiliations YAML into Pandoc MetaInlines
-- so that the default LaTeX article template renders them properly.
-- Affiliations are placed directly below the author names in the title block.

function Meta(meta)
  -- Build affiliation lookup: index -> name
  local affil_map = {}
  if meta.affiliations then
    for _, a in ipairs(meta.affiliations) do
      local idx = pandoc.utils.stringify(a.index)
      local name = pandoc.utils.stringify(a.name)
      affil_map[idx] = name
    end
  end

  -- Build author list with superscript affiliation numbers
  if meta.authors then
    local author_entries = {}
    for _, auth in ipairs(meta.authors) do
      local name = pandoc.utils.stringify(auth.name)
      local affil_idx = ""
      if auth.affiliation then
        affil_idx = pandoc.utils.stringify(auth.affiliation)
      end
      local entry = name
      if affil_idx ~= "" then
        entry = entry .. "\\textsuperscript{" .. affil_idx .. "}"
      end
      if auth.corresponding and pandoc.utils.stringify(auth.corresponding) == "true" then
        entry = entry .. "\\textsuperscript{*}"
      end
      table.insert(author_entries, entry)
    end

    -- Build author line
    local author_str = table.concat(author_entries, ", ")

    -- Build affiliation lines to append below author names
    local affil_lines = {}
    if meta.affiliations then
      for _, a in ipairs(meta.affiliations) do
        local idx = pandoc.utils.stringify(a.index)
        local name = pandoc.utils.stringify(a.name)
        table.insert(affil_lines, "\\textsuperscript{" .. idx .. "}" .. name)
      end
    end

    -- Combine author + affiliations + corresponding note into one block
    local full_block = author_str
    if #affil_lines > 0 then
      full_block = full_block .. " \\\\\n"
        .. "\\vspace{0.3em}\\small " .. table.concat(affil_lines, " \\\\\n\\small ")
      full_block = full_block .. " \\\\\n\\small \\textsuperscript{*}Corresponding author"
    end

    -- Set meta.author so the default Pandoc article template renders it
    meta.author = { pandoc.MetaInlines({ pandoc.RawInline("latex", full_block) }) }
  end

  return meta
end
