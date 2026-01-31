import deepagent_demo as d
from deepagents.middleware import skills as sm

print('skills_dir:', d.skills_dir)
print('skills_root:', d.skills_root)
print('skills_available:', d.skills_available)
print('supports_skills:', d.supports_skills)
print('skills_arg:', d.skills_arg)
print('skills_backend:', type(d.skills_backend).__name__ if d.skills_backend else None)

if d.skills_backend:
    try:
        skills = []
        for path in ("/", ".", ""):
            skills = sm._list_skills(d.skills_backend, path)
            if skills:
                print("skills_source_path:", path)
                break
        print('skills_count:', len(skills))
        for s in skills:
            name = getattr(s, 'name', None) or getattr(s, 'title', None) or str(s)
            slug = getattr(s, 'slug', None)
            desc = getattr(s, 'description', None) or ''
            line = f"- {name}"
            if slug:
                line += f" ({slug})"
            if desc:
                line += f" :: {desc}"
            print(line)
    except Exception as e:
        print('list_skills_error:', type(e).__name__, e)
else:
    print('skills_backend is None; skills not available')
