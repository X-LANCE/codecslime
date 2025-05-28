const generalization_fr_examples = document.querySelectorAll('.generalization_fr');
const cross_page = document.querySelectorAll('.cross_page');
generalization_fr_examples[0].classList.add('active');
cross_page[0].classList.add('active');
cross_page.forEach((page) => {
  page.addEventListener('click', () => {
    const pageNumber = page.getAttribute('data-page');

    // Hide all examples
    generalization_fr_examples.forEach((example) => {
      example.classList.remove('active');
    });
    cross_page.forEach((page) => {
      page.classList.remove('active');
    });

    generalization_fr_examples[pageNumber-1].classList.add('active');
    cross_page[pageNumber-1].classList.add('active');

  });
});
