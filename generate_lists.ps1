$folders = @("train", "val", "test")
foreach ($folder in $folders) {
    if (Test-Path "dataset\$folder") {
        $files = Get-ChildItem -Path "dataset\$folder" -File -Recurse
        $lines = @()
        foreach ($f in $files) {
            $relPath = Resolve-Path -Relative $f.FullName
            if ($relPath.StartsWith(".\")) {
                $relPath = $relPath.Substring(2)
            }
            # Remove extension
            $dir = [System.IO.Path]::GetDirectoryName($relPath)
            $nameNoExt = [System.IO.Path]::GetFileNameWithoutExtension($relPath)
            if ([string]::IsNullOrEmpty($dir)) {
                $relPathWithoutExt = $nameNoExt
            } else {
                $relPathWithoutExt = "$dir/$nameNoExt"
            }
            $relPathWithoutExt = $relPathWithoutExt -replace '\\', '/'
            $lines += $relPathWithoutExt
        }
        $lines | Out-File -FilePath "${folder}_set.txt" -Encoding utf8
    }
}
