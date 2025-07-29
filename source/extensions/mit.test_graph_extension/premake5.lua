-- Use folder name to build extension name and tag. Version is specified explicitly.
local ext = get_current_extension_info()

project_ext (ext)

-- Link only those files and folders into the extension target directory
repo_build.prebuild_link {
    { "data", ext.target_dir.."/data" },
    { "docs", ext.target_dir.."/docs" },
    { "mit", ext.target_dir.."/mit" },
}

local ogn_codegen_script = os.getenv("KIT_SDK_OGN_CODEGEN") or "C:/Users/armmt/.nvidia-omniverse/pkg/kit-sdk-107.3.0/ogn/scripts/ogn_generate.py"
local ogn_nodes_dir = path.join(ext.root_dir, "mit/test_graph_extension/ogn/nodes/optics")
local cmd = string.format('python "%s" --dir "%s" --output "%s"', ogn_codegen_script, ogn_nodes_dir, ogn_nodes_dir)
print("[PREBUILD] Generating OGN Python for all nodes in: " .. ogn_nodes_dir)
os.execute(cmd)
