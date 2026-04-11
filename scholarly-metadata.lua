-- scholarly-metadata.lua
-- Converts JOSS-style authors/affiliations YAML into Pandoc MetaInlines
-- so that the default LaTeX template renders them properly.

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

    -- Set meta.author as a single RawInline block with all authors
    local author_str = table.concat(author_entries, ", ")
    meta.author = { pandoc.MetaInlines({ pandoc.RawInline("latex", author_str) }) }

    -- Build affiliation footnote block
    local affil_lines = {}
    for _, a in ipairs(meta.affiliations) do
      local idx = pandoc.utils.stringify(a.index)
      local name = pandoc.utils.stringify(a.name)
      table.insert(affil_lines, "\\textsuperscript{" .. idx .. "}" .. name)
    end
    if #affil_lines > 0 then
      local affil_block = table.concat(affil_lines, " \\\\\n")
      affil_block = affil_block .. " \\\\\n\\textsuperscript{*}Corresponding author"

      -- Inject as institute or subtitle depending on template
      -- Using the 'institute' variable which many LaTeX templates support
      meta.institute = { pandoc.MetaInlines({ pandoc.RawInline("latex", affil_block) }) }
    end
  end

  return meta
end
