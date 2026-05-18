(function () {
  const toggle = document.getElementById("toc-toggle");
  const sidebar = document.getElementById("toc-sidebar");
  const links = sidebar ? sidebar.querySelectorAll(".toc-list a") : [];

  if (toggle && sidebar) {
    toggle.addEventListener("click", () => {
      const open = sidebar.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  const sections = [];
  links.forEach((link) => {
    const id = link.getAttribute("href")?.slice(1);
    if (!id) return;
    const el = document.getElementById(id);
    if (el) sections.push({ link, el });
    link.addEventListener("click", () => sidebar?.classList.remove("open"));
  });

  if (!sections.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.classList.remove("active"));
        const match = sections.find((s) => s.el === entry.target);
        match?.link.classList.add("active");
      });
    },
    { rootMargin: "-20% 0px -70% 0px", threshold: 0 }
  );

  sections.forEach((s) => observer.observe(s.el));
})();
