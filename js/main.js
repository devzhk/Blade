/**
 * Blade Project Website - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initCopyButton();
    initSmoothScroll();
    initScrollAnimations();
});

/**
 * Copy BibTeX to clipboard functionality
 */
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex code');
    const copyBtn = document.querySelector('.copy-btn');
    
    if (!bibtexCode || !copyBtn) return;
    
    const text = bibtexCode.textContent;
    
    navigator.clipboard.writeText(text).then(function() {
        // Success feedback
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.classList.add('copied');
        
        setTimeout(function() {
            copyBtn.innerHTML = originalHTML;
            copyBtn.classList.remove('copied');
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        // Fallback for older browsers
        fallbackCopy(text);
    });
}

/**
 * Fallback copy method for browsers without clipboard API
 */
function fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    
    try {
        document.execCommand('copy');
        const copyBtn = document.querySelector('.copy-btn');
        if (copyBtn) {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.classList.add('copied');
            
            setTimeout(function() {
                copyBtn.innerHTML = originalHTML;
                copyBtn.classList.remove('copied');
            }, 2000);
        }
    } catch (err) {
        console.error('Fallback copy failed: ', err);
    }
    
    document.body.removeChild(textarea);
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Initialize copy button event listener
 */
function initCopyButton() {
    const copyBtn = document.querySelector('.copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', copyBibtex);
    }
}

/**
 * Initialize scroll-based animations using Intersection Observer
 */
function initScrollAnimations() {
    // Check if IntersectionObserver is supported
    if (!('IntersectionObserver' in window)) return;
    
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe sections for scroll animations
    document.querySelectorAll('.section').forEach(function(section) {
        section.classList.add('animate-on-scroll');
        observer.observe(section);
    });
}

// Add CSS for scroll animations dynamically
(function() {
    const style = document.createElement('style');
    style.textContent = `
        .animate-on-scroll {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.6s ease, transform 0.6s ease;
        }
        .animate-on-scroll.animate-in {
            opacity: 1;
            transform: translateY(0);
        }
    `;
    document.head.appendChild(style);
})();
