using Pkg, LibGit2;
user_regs = joinpath(DEPOT_PATH[1],"registries");
mkpath(user_regs);
all_registries = Dict("General" => "https://github.com/JuliaRegistries/General.git",
	      "HolyLabRegistry" => "https://github.com/HolyLab/HolyLabRegistry.git");
Base.shred!(LibGit2.CachedCredentials()) do creds
  for (reg, url) in all_registries
    path = joinpath(user_regs, reg);
    LibGit2.with(Pkg.GitTools.clone(url, path; header = "registry $reg from $(repr(url))", credentials = creds)) do repo end
  end
end